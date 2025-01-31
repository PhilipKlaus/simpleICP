// From: // https://github.com/rust-ndarray/ndarray/blob/master/examples/sort-axis.rs

use std::cmp::Ordering;
use std::intrinsics::copy_nonoverlapping;
use ndarray::{Array, ArrayBase, Axis, Data, Dimension, RemoveAxis, Zip};
use rawpointer::PointerExt;


// Type invariant: Each index appears exactly once
#[derive(Clone, Debug)]
pub struct Permutation {
    indices: Vec<usize>,
}

impl Permutation {
    /// Checks if the permutation is correct
    pub fn from_indices(v: Vec<usize>) -> Result<Self, ()> {
        let perm = Permutation { indices: v };
        if perm.correct() {
            Ok(perm)
        } else {
            Err(())
        }
    }

    fn correct(&self) -> bool {
        let axis_len = self.indices.len();
        let mut seen = vec![false; axis_len];
        for &i in &self.indices {
            match seen.get_mut(i) {
                None => return false,
                Some(s) => {
                    if *s {
                        return false;
                    } else {
                        *s = true;
                    }
                }
            }
        }
        true
    }
}

pub trait SortArray {
    /// ***Panics*** if `axis` is out of bounds.
    fn identity(&self, axis: Axis) -> Permutation;
    fn sort_axis_by<F>(&self, axis: Axis, less_than: F) -> Permutation
        where
            F: FnMut(usize, usize) -> bool;
}

pub trait PermuteArray {
    type Elem;
    type Dim;
    fn permute_axis(self, axis: Axis, perm: &Permutation) -> Array<Self::Elem, Self::Dim>
        where
            Self::Elem: Clone,
            Self::Dim: RemoveAxis;
}

impl<A, S, D> SortArray for ArrayBase<S, D>
    where
        S: Data<Elem=A>,
        D: Dimension,
{
    fn identity(&self, axis: Axis) -> Permutation {
        Permutation {
            indices: (0..self.len_of(axis)).collect(),
        }
    }

    fn sort_axis_by<F>(&self, axis: Axis, mut less_than: F) -> Permutation
        where
            F: FnMut(usize, usize) -> bool,
    {
        let mut perm = self.identity(axis);
        perm.indices.sort_by(move |&a, &b| {
            if less_than(a, b) {
                Ordering::Less
            } else if less_than(b, a) {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
        perm
    }
}

impl<A, D> PermuteArray for Array<A, D>
    where
        D: Dimension,
{
    type Elem = A;
    type Dim = D;

    fn permute_axis(self, axis: Axis, perm: &Permutation) -> Array<A, D>
        where
            D: RemoveAxis,
    {
        let axis_len = self.len_of(axis);
        let axis_stride = self.stride_of(axis);
        assert_eq!(axis_len, perm.indices.len());
        debug_assert!(perm.correct());

        if self.is_empty() {
            return self;
        }

        let mut result = Array::uninit(self.dim());

        unsafe {
            // logically move ownership of all elements from self into result
            // the result realizes this ownership at .assume_init() further down
            let mut moved_elements = 0;

            // the permutation vector is used like this:
            //
            // index:  0 1 2 3   (index in result)
            // permut: 2 3 0 1   (index in the source)
            //
            // move source 2 -> result 0,
            // move source 3 -> result 1,
            // move source 0 -> result 2,
            // move source 1 -> result 3,
            // et.c.

            let source_0 = self.raw_view().index_axis_move(axis, 0);

            Zip::from(&perm.indices)
                .and(result.axis_iter_mut(axis))
                .for_each(|&perm_i, result_pane| {
                    // Use a shortcut to avoid bounds checking in `index_axis` for the source.
                    //
                    // It works because for any given element pointer in the array we have the
                    // relationship:
                    //
                    // .index_axis(axis, 0) + .stride_of(axis) * j == .index_axis(axis, j)
                    //
                    // where + is pointer arithmetic on the element pointers.
                    //
                    // Here source_0 and the offset is equivalent to self.index_axis(axis, perm_i)
                    Zip::from(result_pane)
                        .and(source_0.clone())
                        .for_each(|to, from_0| {
                            let from = from_0.stride_offset(axis_stride, perm_i);
                            copy_nonoverlapping(from, to.as_mut_ptr(), 1);
                            moved_elements += 1;
                        });
                });
            debug_assert_eq!(result.len(), moved_elements);
            // forget the old elements but not the allocation
            let mut old_storage = self.into_raw_vec();
            old_storage.set_len(0);

            // transfer ownership of the elements into the result
            result.assume_init()
        }
    }
}