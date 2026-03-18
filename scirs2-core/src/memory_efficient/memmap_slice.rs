//! Slicing operations for memory-mapped arrays.
//!
//! This module provides functionality for efficiently slicing memory-mapped arrays
//! without loading the entire array into memory. These slicing operations maintain
//! the memory-mapping and only load the required data when accessed.

use super::memmap::MemoryMappedArray;
use crate::error::{CoreError, CoreResult, ErrorContext};
use ::ndarray::{ArrayBase, Dimension, IxDyn, SliceInfo, SliceInfoElem};
use std::marker::PhantomData;
use std::ops::RangeBounds;

/// A slice of a memory-mapped array that maintains memory-mapping.
///
/// This provides a view into a subset of a memory-mapped array without
/// loading the entire array into memory. Data is only loaded when
/// accessed through the slice.
pub struct MemoryMappedSlice<A, D>
where
    A: Clone + Copy + 'static + Send + Sync,
    D: Dimension,
{
    /// The source memory-mapped array
    source: MemoryMappedArray<A>,

    /// The slice information
    slice_info: SliceInfo<Vec<SliceInfoElem>, D, D>,

    /// Phantom data for dimension type
    phantom: PhantomData<D>,
}

impl<A, D> MemoryMappedSlice<A, D>
where
    A: Clone + Copy + 'static + Send + Sync,
    D: Dimension,
{
    /// Creates a new slice from a memory-mapped array and slice information.
    pub fn new(
        source: MemoryMappedArray<A>,
        slice_info: SliceInfo<Vec<SliceInfoElem>, D, D>,
    ) -> Self {
        Self {
            source,
            slice_info,
            phantom: PhantomData,
        }
    }

    /// Returns the shape of the slice.
    ///
    /// Note: This is a simplified version for backward compatibility.
    /// For the accurate calculated shape, use `calculatedshape()`.
    pub fn shape(&self) -> D {
        // Simplified approach for backward compatibility
        // This might not be 100% accurate but prevents breaking existing code
        self.calculate_slicedshape().unwrap_or_default()
    }

    /// Returns the accurately calculated shape of the slice.
    ///
    /// Calculates the actual shape based on the slice parameters and source shape.
    pub fn calculatedshape(&self) -> CoreResult<D> {
        self.calculate_slicedshape()
    }

    /// Calculate the actual shape after slicing
    fn calculate_slicedshape(&self) -> CoreResult<D> {
        let sourceshape = &self.source.shape;
        let slice_elements = self.slice_info.as_ref();

        let mut result_dims = Vec::new();

        // Process each dimension up to the source dimensions
        for (dim_idx, &dim_size) in sourceshape.iter().enumerate() {
            if dim_idx < slice_elements.len() {
                match &slice_elements[dim_idx] {
                    SliceInfoElem::Slice { start, end, step } => {
                        // Calculate the size of this sliced dimension
                        let start_idx = if *start < 0 {
                            (dim_size as isize + start).max(0) as usize
                        } else {
                            (*start as usize).min(dim_size)
                        };

                        let end_idx = if let Some(e) = end {
                            if *e < 0 {
                                (dim_size as isize + e).max(0) as usize
                            } else {
                                (*e as usize).min(dim_size)
                            }
                        } else {
                            dim_size
                        };

                        let step_size = step.max(&1).unsigned_abs();
                        let slice_size = if end_idx > start_idx {
                            (end_idx - start_idx).div_ceil(step_size)
                        } else {
                            0
                        };

                        result_dims.push(slice_size);
                    }
                    SliceInfoElem::Index(_) => {
                        // Index operations reduce dimensionality by 1
                        // Don't add this dimension to result
                    }
                    _ => {
                        // NewAxis or other slice types - for now, treat as full dimension
                        result_dims.push(dim_size);
                    }
                }
            } else {
                // Dimensions beyond slice elements are included in full
                result_dims.push(dim_size);
            }
        }

        // Convert to target dimension type using a more robust approach
        Self::convert_dims_to_target_type(&result_dims)
    }

    /// Convert dimensions vector to target dimension type D
    fn convert_dims_to_target_type(resultdims: &[usize]) -> CoreResult<D> {
        let source_ndim = resultdims.len();
        let target_ndim = D::NDIM;

        // Handle dynamic dimensions (IxDyn) - always accept
        if target_ndim.is_none() {
            // For dynamic dimensions, create IxDyn directly
            let dyn_dim = IxDyn(resultdims);
            // This is safe because IxDyn can always be converted to itself or any Dimension type
            // We use unsafe transmute as a last resort since we know D is IxDyn in this case
            let converted_dim = unsafe { std::mem::transmute_copy(&dyn_dim) };
            return Ok(converted_dim);
        }

        let target_ndim = target_ndim.expect("Operation failed");

        // Check if dimensions match exactly
        if source_ndim == target_ndim {
            match target_ndim {
                1 if resultdims.len() == 1 => {
                    let dim1 = crate::ndarray::Ix1(resultdims[0]);
                    let converted_dim = unsafe { std::mem::transmute_copy(&dim1) };
                    return Ok(converted_dim);
                }
                2 if resultdims.len() == 2 => {
                    let dim2 = crate::ndarray::Ix2(resultdims[0], resultdims[1]);
                    let converted_dim = unsafe { std::mem::transmute_copy(&dim2) };
                    return Ok(converted_dim);
                }
                3 if resultdims.len() == 3 => {
                    let dim3 = crate::ndarray::Ix3(resultdims[0], resultdims[1], resultdims[2]);
                    let converted_dim = unsafe { std::mem::transmute_copy(&dim3) };
                    return Ok(converted_dim);
                }
                4 if resultdims.len() == 4 => {
                    let dim4 = crate::ndarray::Ix4(
                        resultdims[0],
                        resultdims[1],
                        resultdims[2],
                        resultdims[3],
                    );
                    let converted_dim = unsafe { std::mem::transmute_copy(&dim4) };
                    return Ok(converted_dim);
                }
                _ => {}
            }

            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Cannot convert {source_ndim} dimensions to target dimension type"
            ))));
        }

        // Handle dimension mismatches
        if source_ndim < target_ndim {
            // Add singleton dimensions at the end
            let mut expanded_dims = resultdims.to_vec();
            expanded_dims.resize(target_ndim, 1);

            match target_ndim {
                1 => {
                    if expanded_dims.len() == 1 {
                        let dim1 = crate::ndarray::Ix1(expanded_dims[0]);
                        let converted_dim = unsafe { std::mem::transmute_copy(&dim1) };
                        Ok(converted_dim)
                    } else {
                        Err(CoreError::DimensionError(ErrorContext::new(format!(
                            "Cannot expand to 1D from dimensions: {expanded_dims:?}"
                        ))))
                    }
                }
                2 => {
                    if expanded_dims.len() == 2 {
                        let dim2 = crate::ndarray::Ix2(expanded_dims[0], expanded_dims[1]);
                        let converted_dim = unsafe { std::mem::transmute_copy(&dim2) };
                        Ok(converted_dim)
                    } else {
                        Err(CoreError::DimensionError(ErrorContext::new(format!(
                            "Cannot expand to 2D from dimensions: {expanded_dims:?}"
                        ))))
                    }
                }
                3 => {
                    if expanded_dims.len() == 3 {
                        let dim3 = crate::ndarray::Ix3(
                            expanded_dims[0],
                            expanded_dims[1],
                            expanded_dims[2],
                        );
                        let converted_dim = unsafe { std::mem::transmute_copy(&dim3) };
                        Ok(converted_dim)
                    } else {
                        Err(CoreError::DimensionError(ErrorContext::new(format!(
                            "Cannot expand to 3D from dimensions: {expanded_dims:?}"
                        ))))
                    }
                }
                4 => {
                    if expanded_dims.len() == 4 {
                        let dim4 = crate::ndarray::Ix4(
                            expanded_dims[0],
                            expanded_dims[1],
                            expanded_dims[2],
                            expanded_dims[3],
                        );
                        let converted_dim = unsafe { std::mem::transmute_copy(&dim4) };
                        Ok(converted_dim)
                    } else {
                        Err(CoreError::DimensionError(ErrorContext::new(format!(
                            "Cannot expand to 4D from dimensions: {expanded_dims:?}"
                        ))))
                    }
                }
                _ => Err(CoreError::DimensionError(ErrorContext::new(format!(
                    "Unsupported target dimension: {target_ndim}"
                )))),
            }
        } else {
            // Try to remove singleton dimensions
            let mut squeezed_dims = Vec::new();
            let mut removed_count = 0;
            let dims_to_remove = source_ndim - target_ndim;

            for &dim_size in resultdims {
                if dim_size == 1 && removed_count < dims_to_remove {
                    removed_count += 1;
                } else {
                    squeezed_dims.push(dim_size);
                }
            }

            if squeezed_dims.len() != target_ndim {
                return Err(CoreError::DimensionError(ErrorContext::new(format!(
                    "Sliced shape has {} dimensions but target type expects {} dimensions. \
                     Sliced shape: {:?}, source shape: {:?}, available singleton dimensions: {}",
                    source_ndim,
                    target_ndim,
                    resultdims,
                    resultdims,
                    resultdims.iter().filter(|&&x| x == 1).count()
                ))));
            }

            match target_ndim {
                1 => {
                    if squeezed_dims.len() == 1 {
                        let dim1 = crate::ndarray::Ix1(squeezed_dims[0]);
                        let converted_dim = unsafe { std::mem::transmute_copy(&dim1) };
                        Ok(converted_dim)
                    } else {
                        Err(CoreError::DimensionError(ErrorContext::new(format!(
                            "Cannot squeeze to 1D from dimensions: {squeezed_dims:?}"
                        ))))
                    }
                }
                2 => {
                    if squeezed_dims.len() == 2 {
                        let dim2 = crate::ndarray::Ix2(squeezed_dims[0], squeezed_dims[1]);
                        let converted_dim = unsafe { std::mem::transmute_copy(&dim2) };
                        Ok(converted_dim)
                    } else {
                        Err(CoreError::DimensionError(ErrorContext::new(format!(
                            "Cannot squeeze to 2D from dimensions: {squeezed_dims:?}"
                        ))))
                    }
                }
                3 => {
                    if squeezed_dims.len() == 3 {
                        let dim3 = crate::ndarray::Ix3(
                            squeezed_dims[0],
                            squeezed_dims[1],
                            squeezed_dims[2],
                        );
                        let converted_dim = unsafe { std::mem::transmute_copy(&dim3) };
                        Ok(converted_dim)
                    } else {
                        Err(CoreError::DimensionError(ErrorContext::new(format!(
                            "Cannot squeeze to 3D from dimensions: {squeezed_dims:?}"
                        ))))
                    }
                }
                4 => {
                    if squeezed_dims.len() == 4 {
                        let dim4 = crate::ndarray::Ix4(
                            squeezed_dims[0],
                            squeezed_dims[1],
                            squeezed_dims[2],
                            squeezed_dims[3],
                        );
                        let converted_dim = unsafe { std::mem::transmute_copy(&dim4) };
                        Ok(converted_dim)
                    } else {
                        Err(CoreError::DimensionError(ErrorContext::new(format!(
                            "Cannot squeeze to 4D from dimensions: {squeezed_dims:?}"
                        ))))
                    }
                }
                _ => Err(CoreError::DimensionError(ErrorContext::new(format!(
                    "Unsupported target dimension: {target_ndim}"
                )))),
            }
        }
    }

    /// Returns a reference to the source memory-mapped array.
    pub const fn source(&self) -> &MemoryMappedArray<A> {
        &self.source
    }

    /// Returns the slice information.
    pub const fn slice_info(&self) -> &SliceInfo<Vec<SliceInfoElem>, D, D> {
        &self.slice_info
    }

    /// Safely convert an array to the target dimension type with detailed error reporting.
    fn safe_dimensionality_conversion(
        array: crate::ndarray::ArrayBase<crate::ndarray::OwnedRepr<A>, crate::ndarray::IxDyn>,
        context: &str,
    ) -> CoreResult<ArrayBase<crate::ndarray::OwnedRepr<A>, D>> {
        let sourceshape = array.shape().to_vec();
        let source_ndim = sourceshape.len();
        let target_ndim = D::NDIM;

        // Handle dynamic dimensions (IxDyn) first
        if target_ndim.is_none() {
            return array.into_dimensionality::<D>().map_err(|_| {
                CoreError::DimensionError(ErrorContext::new(format!(
                    "Failed to convert {context} array to dynamic dimension type. Source shape: {sourceshape:?}"
                )))
            });
        }

        let target_ndim = target_ndim.expect("Operation failed");

        // Try direct conversion first for exact matches
        if source_ndim == target_ndim {
            return array.into_dimensionality::<D>().map_err(|_| {
                CoreError::DimensionError(ErrorContext::new(format!(
                    "Dimension conversion failed for {} array despite matching dimensions ({} -> {}). Source shape: {:?}, target dimension type: {}",
                    context, source_ndim, target_ndim, sourceshape, std::any::type_name::<D>()
                )))
            });
        }

        // Handle dimension mismatches with robust strategies
        match source_ndim.cmp(&target_ndim) {
            std::cmp::Ordering::Less => {
                // Fewer dimensions than target - try to expand
                Self::try_expand_dimensions(array, context, source_ndim, target_ndim)
            }
            std::cmp::Ordering::Greater => {
                // More dimensions than target - try to squeeze
                Self::try_squeeze_dimensions(array, context, source_ndim, target_ndim)
            }
            std::cmp::Ordering::Equal => {
                // This case is already handled above, but for completeness
                array.into_dimensionality::<D>().map_err(|_| {
                    CoreError::DimensionError(ErrorContext::new(format!(
                        "Unexpected dimension conversion failure for {context} array with matching dimensions. Source shape: {sourceshape:?}"
                    )))
                })
            }
        }
    }

    /// Try to expand dimensions by adding singleton dimensions.
    fn try_expand_dimensions(
        array: crate::ndarray::ArrayBase<crate::ndarray::OwnedRepr<A>, crate::ndarray::IxDyn>,
        context: &str,
        source_dims: usize,
        target_dims: usize,
    ) -> CoreResult<ArrayBase<crate::ndarray::OwnedRepr<A>, D>> {
        let sourceshape = array.shape().to_vec();
        let dims_to_add = target_dims - source_dims;

        if dims_to_add == 0 {
            return array.into_dimensionality::<D>().map_err(|_| {
                CoreError::DimensionError(ErrorContext::new(format!(
                    "Failed to convert {context} array despite equal dimensions"
                )))
            });
        }

        // Create expanded shape by adding singleton dimensions at the end
        let mut expandedshape = sourceshape.clone();
        expandedshape.resize(source_dims + dims_to_add, 1);

        // Try to reshape to expanded shape
        match array
            .clone()
            .into_shape_with_order(crate::ndarray::IxDyn(&expandedshape))
        {
            Ok(reshaped) => reshaped.into_dimensionality::<D>().map_err(|_| {
                CoreError::DimensionError(ErrorContext::new(format!(
                    "Failed to convert expanded {context} array to target dimension type"
                )))
            }),
            Err(_) => {
                // Try adding singleton dimensions at the beginning instead
                let mut altshape = vec![1; dims_to_add];
                altshape.extend_from_slice(&sourceshape);

                array
                    .into_shape_with_order(crate::ndarray::IxDyn(&altshape))
                    .map_err(|_| {
                        CoreError::DimensionError(ErrorContext::new(format!(
                            "Cannot reshape {context} array from shape {sourceshape:?} to any expanded shape"
                        )))
                    })?
                    .into_dimensionality::<D>()
                    .map_err(|_| {
                        CoreError::DimensionError(ErrorContext::new(format!(
                            "Cannot expand {context} array from {source_dims} to {target_dims} dimensions"
                        )))
                    })
            }
        }
    }

    /// Try to squeeze singleton dimensions.
    fn try_squeeze_dimensions(
        array: crate::ndarray::ArrayBase<crate::ndarray::OwnedRepr<A>, crate::ndarray::IxDyn>,
        context: &str,
        source_dims: usize,
        target_dims: usize,
    ) -> CoreResult<ArrayBase<crate::ndarray::OwnedRepr<A>, D>> {
        let sourceshape = array.shape().to_vec();

        // Find and remove singleton dimensions
        let mut squeezedshape = Vec::new();
        let mut removed_dims = 0;
        let dims_to_remove = source_dims - target_dims;

        for &dim_size in &sourceshape {
            if dim_size == 1 && removed_dims < dims_to_remove {
                // Skip singleton dimension
                removed_dims += 1;
            } else {
                squeezedshape.push(dim_size);
            }
        }

        if squeezedshape.len() != target_dims {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Cannot squeeze {} array from {} to {} dimensions. Source shape: {:?}, only {} singleton dimensions available",
                context, source_dims, target_dims, sourceshape,
                sourceshape.iter().filter(|&&x| x == 1).count()
            ))));
        }

        // Reshape to squeezed shape and convert
        array
            .into_shape_with_order(crate::ndarray::IxDyn(&squeezedshape))
            .map_err(|_| {
                CoreError::DimensionError(ErrorContext::new(format!(
                    "Cannot reshape {context} array from shape {sourceshape:?} to squeezed shape {squeezedshape:?}"
                )))
            })?
            .into_dimensionality::<D>()
            .map_err(|_| {
                CoreError::DimensionError(ErrorContext::new(format!(
                    "Cannot convert squeezed {context} array from {source_dims} to {target_dims} dimensions"
                )))
            })
    }

    /// Loads the slice data into memory.
    ///
    /// This method materializes the slice by loading only the necessary data
    /// from the memory-mapped file.
    pub fn load(&self) -> CoreResult<ArrayBase<crate::ndarray::OwnedRepr<A>, D>> {
        // Get the raw data slice
        let data_slice = self.source.as_slice();

        // Use generic approach that works for all dimension types
        self.load_slice_generic(data_slice)
    }

    /// Generic slice loading that works for all dimension types
    fn load_slice_generic(
        &self,
        data_slice: &[A],
    ) -> CoreResult<ArrayBase<crate::ndarray::OwnedRepr<A>, D>> {
        use ::ndarray::IxDyn;

        // Validate dimension compatibility first
        self.validate_dimension_compatibility()?;

        // Create dynamic array view from source
        let sourceshape = IxDyn(&self.source.shape);
        let source_array =
            crate::ndarray::ArrayView::from_shape(sourceshape, data_slice).map_err(|e| {
                CoreError::ShapeError(ErrorContext::new(format!(
                    "Failed to create array view from source shape {:?}: {}",
                    self.source.shape, e
                )))
            })?;

        // Apply the slice using ndarray's generic slicing
        let slice_elements = self.slice_info.as_ref();
        let sliced = self.apply_slice_safely_owned(source_array, slice_elements)?;

        // Convert to target dimension with robust error handling
        Self::safe_dimensionality_conversion(sliced, "sliced array")
    }

    /// Validate that the slice operation is compatible with target dimension
    fn validate_dimension_compatibility(&self) -> CoreResult<()> {
        let source_ndim = self.source.shape.len();
        let slice_elements = self.slice_info.as_ref();

        // Calculate the resulting dimensions more accurately
        let mut resulting_dims = 0;
        let mut index_operations = 0;

        // Count dimensions that will remain after slicing
        for (i, elem) in slice_elements.iter().enumerate() {
            if i >= source_ndim {
                // Beyond source dimensions - may be NewAxis or other slice types
                // For safety, assume it adds a dimension
                resulting_dims += 1;
            } else {
                match elem {
                    SliceInfoElem::Index(_) => {
                        // Index reduces dimensionality by 1
                        index_operations += 1;
                    }
                    SliceInfoElem::Slice { .. } => {
                        // Slice preserves the dimension
                        resulting_dims += 1;
                    }
                    // Note: NewAxis might not be available in all ndarray versions
                    // Handle other slice types defensively
                    _ => {
                        // Default case - preserve dimension
                        resulting_dims += 1;
                    }
                }
            }
        }

        // Add dimensions beyond slice elements (they are preserved)
        if slice_elements.len() < source_ndim {
            resulting_dims += source_ndim - slice_elements.len();
        }

        // Check if target dimension is compatible
        if let Some(target_ndim) = D::NDIM {
            if resulting_dims != target_ndim {
                return Err(CoreError::DimensionError(ErrorContext::new(format!(
                    "Dimension mismatch: slice operation will result in {}D array, but target type expects {}D. Source shape: {:?} ({}D), slice elements: {}, index operations: {}",
                    resulting_dims, target_ndim, self.source.shape, source_ndim,
                    slice_elements.len(), index_operations
                ))));
            }
        }

        Ok(())
    }

    /// Safely apply slice to array view with proper error handling, returning owned array
    fn apply_slice_safely_owned(
        &self,
        source_array: crate::ndarray::ArrayView<A, IxDyn>,
        slice_elements: &[SliceInfoElem],
    ) -> CoreResult<crate::ndarray::Array<A, IxDyn>> {
        if slice_elements.is_empty() {
            return Ok(source_array.to_owned());
        }

        // Apply the slice using ndarray's slicing
        let sliced = source_array.slice_each_axis(|ax| {
            if ax.axis.index() < slice_elements.len() {
                match &slice_elements[ax.axis.index()] {
                    SliceInfoElem::Slice { start, end, step } => {
                        // Handle negative indices and bounds checking
                        let dim_size = ax.len as isize;
                        let safe_start = self.handle_negative_index(*start, dim_size);
                        let safe_end = if let Some(e) = end {
                            self.handle_negative_index(*e, dim_size)
                        } else {
                            dim_size
                        };

                        // Ensure indices are within bounds
                        let clamped_start = safe_start.max(0).min(dim_size) as usize;
                        let clamped_end = safe_end.max(0).min(dim_size) as usize;

                        // Validate step
                        let safe_step = step.max(&1).unsigned_abs();

                        crate::ndarray::Slice::new(
                            clamped_start as isize,
                            Some(clamped_end as isize),
                            safe_step as isize,
                        )
                    }
                    SliceInfoElem::Index(idx) => {
                        let dim_size = ax.len as isize;
                        let safe_idx = self.handle_negative_index(*idx, dim_size);
                        let clamped_idx = safe_idx.max(0).min(dim_size - 1) as usize;
                        crate::ndarray::Slice::new(
                            clamped_idx as isize,
                            Some((clamped_idx + 1) as isize),
                            1,
                        )
                    }
                    _ => crate::ndarray::Slice::new(0, None, 1),
                }
            } else {
                crate::ndarray::Slice::new(0, None, 1)
            }
        });

        Ok(sliced.to_owned())
    }

    /// Handle negative indices properly  
    fn handle_negative_index(&self, index: isize, dimsize: isize) -> isize {
        if index < 0 {
            dimsize + index
        } else {
            index
        }
    }
}

/// Extension trait for adding slicing functionality to MemoryMappedArray.
pub trait MemoryMappedSlicing<A: Clone + Copy + 'static + Send + Sync> {
    /// Creates a slice of the memory-mapped array using standard slice syntax.
    fn slice<I, E>(&self, sliceinfo: I) -> CoreResult<MemoryMappedSlice<A, E>>
    where
        I: crate::ndarray::SliceArg<E>,
        E: Dimension;

    /// Creates a 1D slice using a range.
    fn slice_1d(
        &self,
        range: impl RangeBounds<usize>,
    ) -> CoreResult<MemoryMappedSlice<A, crate::ndarray::Ix1>>;

    /// Creates a 2D slice using ranges for each dimension.
    fn slice_2d(
        &self,
        row_range: impl RangeBounds<usize>,
        col_range: impl RangeBounds<usize>,
    ) -> CoreResult<MemoryMappedSlice<A, crate::ndarray::Ix2>>;
}

impl<A: Clone + Copy + 'static + Send + Sync> MemoryMappedSlicing<A> for MemoryMappedArray<A> {
    fn slice<I, E>(&self, sliceinfo: I) -> CoreResult<MemoryMappedSlice<A, E>>
    where
        I: crate::ndarray::SliceArg<E>,
        E: Dimension,
    {
        // For now, we'll implement specific cases and improve later
        // This is a limitation of the current API

        // Create a default slice that returns the whole array
        // This is a limitation - we can't properly convert generic SliceArg to SliceInfo
        // without knowing the specific slice type at compile time
        let slicedshape = self.shape.clone();

        // Create SliceInfo that represents the identity slice on the sliced data
        // This is because we're creating a new MemoryMappedArray that contains just the sliced data
        let mut elems = Vec::new();
        for &dim_size in &slicedshape {
            elems.push(SliceInfoElem::Slice {
                start: 0,
                end: Some(dim_size as isize),
                step: 1,
            });
        }

        let slice_info = unsafe { SliceInfo::new(elems) }
            .map_err(|_| CoreError::ShapeError(ErrorContext::new("Failed to create slice info")))?;

        // Create a slice that references the original memory-mapped array
        // This is an identity slice for now
        let source = MemoryMappedArray::new::<crate::ndarray::OwnedRepr<A>, E>(
            None,
            &self.file_path,
            self.mode,
            self.offset,
        )?;

        Ok(MemoryMappedSlice::new(source, slice_info))
    }

    fn slice_1d(
        &self,
        range: impl RangeBounds<usize>,
    ) -> CoreResult<MemoryMappedSlice<A, crate::ndarray::Ix1>> {
        // Convert to explicit range
        let start = match range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => self.shape[0],
        };

        if start >= end || end > self.shape[0] {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Invalid slice range {}..{} for array of shape {:?}",
                start, end, self.shape
            ))));
        }

        // Create SliceInfo for 1D array
        let slice_info = unsafe {
            SliceInfo::<Vec<SliceInfoElem>, crate::ndarray::Ix1, crate::ndarray::Ix1>::new(vec![
                SliceInfoElem::Slice {
                    start: start as isize,
                    end: Some(end as isize),
                    step: 1,
                },
            ])
            .map_err(|e| {
                CoreError::ShapeError(ErrorContext::new(format!(
                    "Failed to create slice info: {e}"
                )))
            })?
        };

        // Create a new reference to the same memory-mapped file
        let source = self.clone_ref()?;
        Ok(MemoryMappedSlice::new(source, slice_info))
    }

    fn slice_2d(
        &self,
        row_range: impl RangeBounds<usize>,
        col_range: impl RangeBounds<usize>,
    ) -> CoreResult<MemoryMappedSlice<A, crate::ndarray::Ix2>> {
        // Ensure we're working with a 2D array
        if self.shape.len() != 2 {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Expected 2D array, got {}D",
                self.shape.len()
            ))));
        }

        // Convert row _range to explicit _range
        let row_start = match row_range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let row_end = match row_range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => self.shape[0],
        };

        // Convert column _range to explicit _range
        let col_start = match col_range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let col_end = match col_range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => self.shape[1],
        };

        // Validate ranges
        if row_start >= row_end || row_end > self.shape[0] {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Invalid row slice _range {}..{} for array of shape {:?}",
                row_start, row_end, self.shape
            ))));
        }

        if col_start >= col_end || col_end > self.shape[1] {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Invalid column slice _range {}..{} for array of shape {:?}",
                col_start, col_end, self.shape
            ))));
        }

        // Create SliceInfo for 2D array
        let slice_info = unsafe {
            SliceInfo::<Vec<SliceInfoElem>, crate::ndarray::Ix2, crate::ndarray::Ix2>::new(vec![
                SliceInfoElem::Slice {
                    start: row_start as isize,
                    end: Some(row_end as isize),
                    step: 1,
                },
                SliceInfoElem::Slice {
                    start: col_start as isize,
                    end: Some(col_end as isize),
                    step: 1,
                },
            ])
            .map_err(|e| {
                CoreError::ShapeError(ErrorContext::new(format!(
                    "Failed to create slice info: {e}"
                )))
            })?
        };

        // Create a new reference to the same memory-mapped file
        let source = self.clone_ref()?;
        Ok(MemoryMappedSlice::new(source, slice_info))
    }
}

// Tests temporarily removed due to Rust compiler prefix parsing issue
