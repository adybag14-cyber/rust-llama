//! LP64 CBLAS matrix adapter. Vendor CBLAS is an explicitly external boundary.
//! No original ggml implementation is linked. Dimensions are checked before FFI.
use crate::{Error, Result};
use llama_platform_loader::DynamicLibrary;
use std::ffi::c_int;

/// A checked two-dimensional view. Strides and offset are in f32 elements.
#[derive(Clone, Copy, Debug)]
pub struct MatrixView<'a> {
    data: &'a [f32],
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
    pub col_stride: usize,
    offset: usize,
}
impl<'a> MatrixView<'a> {
    pub fn new(
        data: &'a [f32],
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,
        offset: usize,
    ) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(Error::Bounds("matrix dimensions must be positive"));
        }
        if row_stride == 0 || col_stride == 0 {
            return Err(Error::Bounds("matrix strides must be positive"));
        }
        let end = (rows - 1)
            .checked_mul(row_stride)
            .and_then(|n| {
                (cols - 1)
                    .checked_mul(col_stride)
                    .and_then(|m| n.checked_add(m))
            })
            .and_then(|n| n.checked_add(offset))
            .ok_or(Error::Bounds("matrix address overflow"))?;
        if end >= data.len() {
            return Err(Error::Bounds("matrix view exceeds storage"));
        }
        Ok(Self {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
            offset,
        })
    }
    pub fn contiguous(data: &'a [f32], rows: usize, cols: usize) -> Result<Self> {
        Self::new(data, rows, cols, cols, 1, 0)
    }
    pub fn transpose(self) -> Self {
        Self {
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
            ..self
        }
    }
    pub fn at(self, row: usize, col: usize) -> Result<f32> {
        if row >= self.rows || col >= self.cols {
            return Err(Error::Bounds("matrix index"));
        }
        Ok(self.data[self.offset + row * self.row_stride + col * self.col_stride])
    }
    fn packed(self) -> Result<Vec<f32>> {
        let len = self
            .rows
            .checked_mul(self.cols)
            .ok_or(Error::Bounds("matrix packing overflow"))?;
        let mut data = Vec::new();
        data.try_reserve_exact(len)
            .map_err(|_| Error::Bounds("matrix packing allocation failed"))?;
        for i in 0..self.rows {
            for j in 0..self.cols {
                data.push(self.data[self.offset + i * self.row_stride + j * self.col_stride]);
            }
        }
        Ok(data)
    }
}

type Sgemm = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    f32,
    *mut f32,
    c_int,
);

/// An owned LP64 CBLAS provider. Deliberately not Send/Sync: no assertion about
/// a provider's thread-safety is made here; no global vendor thread count changes.
pub struct Blas {
    library: DynamicLibrary,
    sgemm: Sgemm,
}
impl Blas {
    pub fn open() -> Result<Self> {
        #[cfg(target_os = "linux")]
        let names = ["libblas.so.3", "libopenblas.so.0"];
        #[cfg(target_os = "macos")]
        let names = ["/System/Library/Frameworks/Accelerate.framework/Accelerate"];
        #[cfg(target_os = "windows")]
        let names = ["libopenblas.dll"];
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        let names: [&str; 0] = [];
        let library = DynamicLibrary::open(&names)?;
        // SAFETY: these candidates use the LP64 cblas_sgemm C signature. The
        // module handle is owned by this struct and outlives every invocation.
        let sgemm = unsafe { library.symbol::<Sgemm>(c"cblas_sgemm") }?;
        Ok(Self { library, sgemm })
    }
    pub fn provider(&self) -> &str {
        self.library.name()
    }

    /// C = A * B, row-major output; padded/strided views are packed in Rust.
    pub fn matmul(&self, a: MatrixView<'_>, b: MatrixView<'_>) -> Result<Vec<f32>> {
        if a.cols != b.rows {
            return Err(Error::Bounds("matmul inner dimension mismatch"));
        }
        let m =
            c_int::try_from(a.rows).map_err(|_| Error::Bounds("CBLAS m exceeds LP64 integer"))?;
        let n =
            c_int::try_from(b.cols).map_err(|_| Error::Bounds("CBLAS n exceeds LP64 integer"))?;
        let k =
            c_int::try_from(a.cols).map_err(|_| Error::Bounds("CBLAS k exceeds LP64 integer"))?;
        let size = a
            .rows
            .checked_mul(b.cols)
            .ok_or(Error::Bounds("matmul output overflow"))?;
        let mut c = Vec::new();
        c.try_reserve_exact(size)
            .map_err(|_| Error::Bounds("matmul output allocation failed"))?;
        c.resize(size, 0.0);
        // Packing permits arbitrary checked views without passing strides that
        // CBLAS cannot express and without handing it an aliased output slice.
        let aa = a.packed()?;
        let bb = b.packed()?;
        // SAFETY: all dimensions fit c_int; aa/bb/c contain m*k/k*n/m*n
        // contiguous f32 values; output does not alias either input; handles live.
        unsafe {
            (self.sgemm)(
                101,
                111,
                111,
                m,
                n,
                k,
                1.0,
                aa.as_ptr(),
                k,
                bb.as_ptr(),
                n,
                0.0,
                c.as_mut_ptr(),
                n,
            );
        }
        Ok(c)
    }
    /// GGML MUL_MAT shape convention: weights [K,M], inputs [K,N], result [M,N],
    /// where dimension zero is contiguous. MatrixView uses [row,column], so
    /// weights is M x K and inputs N x K, yielding row-major N x M.
    pub fn ggml_mul_mat(
        &self,
        weights: MatrixView<'_>,
        inputs: MatrixView<'_>,
    ) -> Result<Vec<f32>> {
        self.matmul(inputs, weights.transpose())
    }
    /// F32 outer-product reduction, matching the snapshot's two-dimensional path.
    pub fn ggml_out_prod(&self, left: MatrixView<'_>, right: MatrixView<'_>) -> Result<Vec<f32>> {
        if left.rows != right.rows {
            return Err(Error::Bounds("out-product reduction mismatch"));
        }
        self.matmul(right.transpose(), left)
    }
}

/// Snapshot ggml-blas.cpp's size/contiguity/Hadamard preference, restricted to
/// the implemented F32 path. Quantized conversion and four-dimensional batching
/// remain a T03/T02 integration gate.
pub fn prefer_f32_mul_mat(
    m: usize,
    n: usize,
    k: usize,
    contiguous: bool,
    hadamard_hint: bool,
) -> bool {
    contiguous && !hadamard_hint && m >= 32 && n >= 32 && k >= 32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn view_checks_bounds_and_overflow() {
        assert!(MatrixView::new(&[1.0; 4], 2, 2, usize::MAX, 1, 0).is_err());
        assert!(MatrixView::new(&[1.0; 4], 2, 2, 2, 1, 1).is_err());
        assert!(MatrixView::contiguous(&[], 0, 0).is_err());
        assert!(MatrixView::new(&[1.0; 4], 2, 2, 0, 1, 0).is_err());
        assert!(
            MatrixView::contiguous(&[1.0; 4], 2, 2)
                .unwrap()
                .at(2, 0)
                .is_err()
        );
    }
    #[test]
    fn snapshot_dispatch_threshold_and_hint_are_preserved() {
        assert!(prefer_f32_mul_mat(32, 32, 32, true, false));
        for (m, n, k, c, h) in [
            (31, 32, 32, true, false),
            (32, 31, 32, true, false),
            (32, 32, 31, true, false),
            (32, 32, 32, false, false),
            (32, 32, 32, true, true),
        ] {
            assert!(!prefer_f32_mul_mat(m, n, k, c, h));
        }
    }
    #[test]
    #[ignore = "requires an installed LP64 CBLAS provider"]
    fn external_blas_matmul_and_strides() {
        let blas =
            Blas::open().expect("install an LP64 CBLAS provider before running this ignored test");
        let a = MatrixView::new(&[99., 1., 2., 3., 99., 4., 5., 6., 99.], 2, 3, 4, 1, 1).unwrap();
        let b = MatrixView::contiguous(&[7., 8., 9., 10., 11., 12.], 3, 2).unwrap();
        assert_eq!(blas.matmul(a, b).unwrap(), vec![58., 64., 139., 154.]);
        let wt = MatrixView::contiguous(&[1., 2., 3., 4., 5., 6.], 2, 3).unwrap();
        let xs = MatrixView::contiguous(&[7., 9., 11., 8., 10., 12.], 2, 3).unwrap();
        assert_eq!(
            blas.ggml_mul_mat(wt, xs).unwrap(),
            vec![58., 139., 64., 154.]
        );
        assert!(blas.matmul(a, a).is_err());
    }
    #[test]
    #[ignore = "requires an installed LP64 CBLAS provider"]
    fn external_blas_outer_product_reduction() {
        let blas =
            Blas::open().expect("install an LP64 CBLAS provider before running this ignored test");
        let left = MatrixView::contiguous(&[1., 2., 3., 4., 5., 6.], 3, 2).unwrap();
        let right = MatrixView::contiguous(&[2., 3., 4.], 3, 1).unwrap();
        assert_eq!(blas.ggml_out_prod(left, right).unwrap(), vec![31., 40.]);
    }
}
