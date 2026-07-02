//! Transform composition for building pipelines.

use super::lazy::{materialize_ops, push_pending, LazyTransform, PendingOp};
use crate::error::Result;
use crate::nifti::NiftiImage;
use std::borrow::Cow;

/// A composable transform that can be applied eagerly or lazily.
pub trait Transform {
    /// Apply the transform eagerly to an image.
    ///
    /// # Errors
    /// Returns an error if the transform fails (e.g., non-contiguous array,
    /// memory allocation failure, or invalid data).
    fn apply(&self, image: &NiftiImage) -> Result<NiftiImage>;
}

/// A composed pipeline of transforms.
pub struct Compose {
    transforms: Vec<Box<dyn TransformBox>>,
    lazy: bool,
}

/// Internal trait for type-erased transforms.
trait TransformBox: Send + Sync {
    fn apply_eager(&self, image: &NiftiImage) -> Result<NiftiImage>;
    fn to_pending(&self, image: &NiftiImage) -> Option<Vec<PendingOp>>;
    fn requires_data(&self) -> bool;
}

impl<T: Transform + LazyTransform + Send + Sync + 'static> TransformBox for T {
    fn apply_eager(&self, image: &NiftiImage) -> Result<NiftiImage> {
        self.apply(image)
    }

    fn to_pending(&self, image: &NiftiImage) -> Option<Vec<PendingOp>> {
        self.to_pending_op(image)
    }

    fn requires_data(&self) -> bool {
        LazyTransform::requires_data(self)
    }
}

impl Compose {
    /// Create a new empty composition.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            lazy: true,
        }
    }

    /// Create a composition with lazy evaluation disabled.
    pub fn eager() -> Self {
        Self {
            transforms: Vec::new(),
            lazy: false,
        }
    }

    /// Add a transform to the pipeline.
    pub fn push<T: Transform + LazyTransform + Send + Sync + 'static>(
        mut self,
        transform: T,
    ) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }

    /// Enable or disable lazy evaluation.
    pub fn lazy(mut self, lazy: bool) -> Self {
        self.lazy = lazy;
        self
    }

    /// Apply the composed transforms to an image.
    ///
    /// # Errors
    /// Returns `Err` if any transform fails (I/O error, invalid data, etc.)
    pub fn apply(&self, image: &NiftiImage) -> Result<NiftiImage> {
        if self.lazy {
            self.apply_lazy(image)
        } else {
            self.apply_eager(image)
        }
    }

    /// Apply transforms eagerly (one at a time).
    fn apply_eager(&self, image: &NiftiImage) -> Result<NiftiImage> {
        let mut iter = self.transforms.iter();
        let Some(first) = iter.next() else {
            return Ok(image.clone());
        };
        let mut result = first.apply_eager(image)?;
        for transform in iter {
            result = transform.apply_eager(&result)?;
        }
        Ok(result)
    }

    /// Apply transforms lazily (accumulate and execute in batches).
    ///
    /// The input is borrowed until a transform produces a new image or pending
    /// ops must be materialized, so the common all-lazy pipeline never clones
    /// the input volume.
    fn apply_lazy(&self, image: &NiftiImage) -> Result<NiftiImage> {
        if self.transforms.is_empty() {
            return Ok(image.clone());
        }

        let mut base: Cow<'_, NiftiImage> = Cow::Borrowed(image);
        let mut pending: Vec<PendingOp> = Vec::new();

        for transform in &self.transforms {
            if transform.requires_data() {
                if !pending.is_empty() {
                    base = Cow::Owned(materialize_ops(base.as_ref(), &pending)?);
                    pending.clear();
                }
                base = Cow::Owned(transform.apply_eager(base.as_ref())?);
            } else if let Some(ops) = transform.to_pending(base.as_ref()) {
                for op in ops {
                    push_pending(&mut pending, op);
                }
            } else {
                let materialized = materialize_ops(base.as_ref(), &pending)?;
                base = Cow::Owned(transform.apply_eager(&materialized)?);
                pending.clear();
            }
        }

        if pending.is_empty() {
            Ok(base.into_owned())
        } else {
            materialize_ops(base.as_ref(), &pending)
        }
    }

    /// Get the number of transforms in the pipeline.
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Check if the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl Default for Compose {
    fn default() -> Self {
        Self::new()
    }
}

/// A builder for transform pipelines with a fluent API.
pub struct TransformPipeline {
    compose: Compose,
}

impl TransformPipeline {
    /// Create a new transform pipeline.
    pub fn new() -> Self {
        Self {
            compose: Compose::new(),
        }
    }

    /// Add z-normalization to the pipeline.
    pub fn z_normalize(self) -> Self {
        Self {
            compose: self.compose.push(ZNormalizeTransform),
        }
    }

    /// Add intensity rescaling to the pipeline.
    pub fn rescale(self, out_min: f32, out_max: f32) -> Self {
        Self {
            compose: self.compose.push(RescaleTransform { out_min, out_max }),
        }
    }

    /// Add clamping to the pipeline.
    pub fn clamp(self, min: f32, max: f32) -> Self {
        Self {
            compose: self.compose.push(ClampTransform { min, max }),
        }
    }

    /// Add resampling to target spacing.
    pub fn resample_to_spacing(self, spacing: [f32; 3]) -> Self {
        Self {
            compose: self.compose.push(ResampleSpacingTransform { spacing }),
        }
    }

    /// Add resampling to target shape.
    pub fn resample_to_shape(self, shape: [usize; 3]) -> Self {
        Self {
            compose: self.compose.push(ResampleShapeTransform { shape }),
        }
    }

    /// Add axis flipping.
    pub fn flip(self, axes: &[usize]) -> Self {
        let mut mask = 0u8;
        for &axis in axes {
            mask |= 1 << axis;
        }
        Self {
            compose: self.compose.push(FlipTransform { axes: mask }),
        }
    }

    /// Enable or disable lazy evaluation.
    pub fn lazy(mut self, lazy: bool) -> Self {
        self.compose = self.compose.lazy(lazy);
        self
    }

    /// Apply the pipeline to an image.
    ///
    /// # Errors
    /// Returns `Err` if any transform fails (I/O error, invalid data, etc.)
    pub fn apply(&self, image: &NiftiImage) -> Result<NiftiImage> {
        self.compose.apply(image)
    }

    /// Get the underlying Compose.
    pub fn into_compose(self) -> Compose {
        self.compose
    }
}

impl Default for TransformPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// Built-in transform implementations

struct ZNormalizeTransform;

impl Transform for ZNormalizeTransform {
    fn apply(&self, image: &NiftiImage) -> Result<NiftiImage> {
        crate::transforms::z_normalization(image)
    }
}

impl LazyTransform for ZNormalizeTransform {
    fn to_pending_op(&self, _image: &NiftiImage) -> Option<Vec<PendingOp>> {
        // Stats are deferred to materialization, letting z-normalize fuse with a
        // following clamp or linear op in one pass.
        Some(vec![PendingOp::ZNormalize])
    }
}

struct RescaleTransform {
    out_min: f32,
    out_max: f32,
}

impl Transform for RescaleTransform {
    fn apply(&self, image: &NiftiImage) -> Result<NiftiImage> {
        crate::transforms::rescale_intensity(image, self.out_min as f64, self.out_max as f64)
    }
}

impl LazyTransform for RescaleTransform {
    fn to_pending_op(&self, _image: &NiftiImage) -> Option<Vec<PendingOp>> {
        // Rescaling requires knowing min/max, needs data
        None
    }

    fn requires_data(&self) -> bool {
        true // Need to compute min/max
    }
}

struct ClampTransform {
    min: f32,
    max: f32,
}

impl Transform for ClampTransform {
    fn apply(&self, image: &NiftiImage) -> Result<NiftiImage> {
        crate::transforms::clamp(image, self.min as f64, self.max as f64)
    }
}

impl LazyTransform for ClampTransform {
    fn to_pending_op(&self, _image: &NiftiImage) -> Option<Vec<PendingOp>> {
        Some(vec![PendingOp::Clamp {
            min: self.min,
            max: self.max,
        }])
    }
}

struct ResampleSpacingTransform {
    spacing: [f32; 3],
}

impl Transform for ResampleSpacingTransform {
    fn apply(&self, image: &NiftiImage) -> Result<NiftiImage> {
        crate::transforms::resample_to_spacing(
            image,
            self.spacing,
            crate::transforms::Interpolation::Trilinear,
        )
    }
}

impl LazyTransform for ResampleSpacingTransform {
    fn to_pending_op(&self, image: &NiftiImage) -> Option<Vec<PendingOp>> {
        // Get current spacing from image
        let current_spacing = image.spacing();

        // Compute scale factors: new_spacing / old_spacing
        // This maps output coordinates to input coordinates
        let scale_x = self.spacing[0] / current_spacing[0];
        let scale_y = self.spacing[1] / current_spacing[1];
        let scale_z = self.spacing[2] / current_spacing[2];

        // Compute output shape
        let shape = image.shape();
        let new_shape = [
            ((shape[0] as f32) / scale_x).round() as usize,
            ((shape[1] as f32) / scale_y).round() as usize,
            ((shape[2] as f32) / scale_z).round() as usize,
        ];

        // Create scaling matrix (maps output voxel coords to input voxel coords)
        let matrix = [
            [scale_x, 0.0, 0.0, 0.0],
            [0.0, scale_y, 0.0, 0.0],
            [0.0, 0.0, scale_z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        Some(vec![PendingOp::Affine {
            matrix,
            output_shape: Some(new_shape),
            interpolation: super::lazy::Interpolation::Trilinear,
        }])
    }

    fn requires_data(&self) -> bool {
        false // Can be lazily composed now
    }
}

struct ResampleShapeTransform {
    shape: [usize; 3],
}

impl Transform for ResampleShapeTransform {
    fn apply(&self, image: &NiftiImage) -> Result<NiftiImage> {
        crate::transforms::resample_to_shape(
            image,
            self.shape,
            crate::transforms::Interpolation::Trilinear,
        )
    }
}

impl LazyTransform for ResampleShapeTransform {
    fn to_pending_op(&self, image: &NiftiImage) -> Option<Vec<PendingOp>> {
        // Get current shape from image
        let current_shape = image.shape();

        // Compute scale factors: old_shape / new_shape
        // This maps output coordinates to input coordinates
        let scale_x = current_shape[0] as f32 / self.shape[0] as f32;
        let scale_y = current_shape[1] as f32 / self.shape[1] as f32;
        let scale_z = current_shape[2] as f32 / self.shape[2] as f32;

        // Create scaling matrix (maps output voxel coords to input voxel coords)
        let matrix = [
            [scale_x, 0.0, 0.0, 0.0],
            [0.0, scale_y, 0.0, 0.0],
            [0.0, 0.0, scale_z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        Some(vec![PendingOp::Affine {
            matrix,
            output_shape: Some(self.shape),
            interpolation: super::lazy::Interpolation::Trilinear,
        }])
    }

    fn requires_data(&self) -> bool {
        false // Can be lazily composed now
    }
}

struct FlipTransform {
    axes: u8,
}

impl Transform for FlipTransform {
    fn apply(&self, image: &NiftiImage) -> Result<NiftiImage> {
        let axes_vec: Vec<usize> = (0..3).filter(|&i| (self.axes >> i) & 1 == 1).collect();
        crate::transforms::flip(image, &axes_vec)
    }
}

impl LazyTransform for FlipTransform {
    fn to_pending_op(&self, _image: &NiftiImage) -> Option<Vec<PendingOp>> {
        Some(vec![PendingOp::Flip { axes: self.axes }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nifti::NiftiImage;
    use ndarray::ArrayD;

    fn create_test_image(data: Vec<f32>, shape: [usize; 3]) -> NiftiImage {
        use ndarray::ShapeBuilder;
        // Create C-order array first, then convert to F-order to match NIfTI convention
        let c_order = ArrayD::from_shape_vec(shape.to_vec(), data).unwrap();
        let mut f_order = ArrayD::zeros(ndarray::IxDyn(&shape).f());
        f_order.assign(&c_order);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        NiftiImage::from_array(f_order, affine)
    }

    #[test]
    fn test_compose_basic() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        let pipeline = Compose::new()
            .push(ZNormalizeTransform)
            .push(ClampTransform {
                min: -2.0,
                max: 2.0,
            });

        let result = pipeline.apply(&img).expect("pipeline should succeed");
        assert_eq!(result.shape(), &[4, 4, 4]);
    }

    #[test]
    fn test_compose_eager_vs_lazy() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        // Test eager execution
        let eager_pipeline = Compose::eager().push(ClampTransform {
            min: 0.0,
            max: 50.0,
        });
        let eager_result = eager_pipeline
            .apply(&img)
            .expect("eager pipeline should succeed");

        // Test lazy execution
        let lazy_pipeline = Compose::new().push(ClampTransform {
            min: 0.0,
            max: 50.0,
        });
        let lazy_result = lazy_pipeline
            .apply(&img)
            .expect("lazy pipeline should succeed");

        // Both should produce same shape
        assert_eq!(eager_result.shape(), lazy_result.shape());
    }

    #[test]
    fn test_transform_pipeline_fluent() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        let pipeline = TransformPipeline::new()
            .z_normalize()
            .clamp(-1.0, 1.0)
            .flip(&[0]);

        let result = pipeline.apply(&img).expect("pipeline should succeed");
        assert_eq!(result.shape(), &[4, 4, 4]);
    }

    #[test]
    fn test_compose_resample_shape() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        let pipeline = TransformPipeline::new().resample_to_shape([8, 8, 8]);

        let result = pipeline.apply(&img).expect("pipeline should succeed");
        assert_eq!(result.shape(), &[8, 8, 8]);
    }

    #[test]
    fn test_compose_chain_resamples() {
        // Test that multiple resamples can be chained
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        let pipeline = TransformPipeline::new()
            .resample_to_shape([8, 8, 8])
            .resample_to_shape([16, 16, 16]);

        let result = pipeline.apply(&img).expect("pipeline should succeed");
        assert_eq!(result.shape(), &[16, 16, 16]);
    }

    #[test]
    fn test_compose_intensity_then_resample() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        let pipeline = TransformPipeline::new()
            .z_normalize()
            .clamp(-1.0, 1.0)
            .resample_to_shape([8, 8, 8]);

        let result = pipeline.apply(&img).expect("pipeline should succeed");
        assert_eq!(result.shape(), &[8, 8, 8]);

        // Check values are clamped
        let data = result.to_f32().unwrap();
        for &v in data.iter() {
            assert!(v >= -1.0 && v <= 1.0, "Value {} outside clamp range", v);
        }
    }

    #[test]
    fn test_compose_is_empty() {
        let empty = Compose::new();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let with_transform = Compose::new().push(ZNormalizeTransform);
        assert!(!with_transform.is_empty());
        assert_eq!(with_transform.len(), 1);
    }
}
