use std::num::FpCategory::Nan;
use ndarray::{Array, Array1, Axis};
use crate::pointcloud::{Item, PointCloud};
use ndarray_stats::QuantileExt;
use typenum::N64;

use ndarray::prelude::*;
use ndarray::{Data, RemoveAxis, Zip};

use rawpointer::PointerExt;

use std::cmp::Ordering;
use std::ptr::copy_nonoverlapping;
use std::time::Instant;

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
        S: Data<Elem = A>,
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

//##################################################################################################

// pc1 == fixed / pc2 == moved
pub fn match_point_clouds(pc1: &PointCloud, pc2: &PointCloud) -> (Array1<f64>, Array1<f64>) {
    let now = Instant::now();

    let pts_sel_pc1 = pc1.get_selected_points();
    let mut pts_sel_pc2 = pc2.get_selected_points();

    let idx_pc1 = pc1.get_idx_of_selected_points();

    let nn = PointCloud::knn_search(&mut pts_sel_pc2, &pts_sel_pc1, 1);
    let mut pc2_nn_pts: Vec<Item> = Vec::with_capacity(nn.len());
    for (idx, nn_res) in nn.iter().enumerate() {
        pc2_nn_pts.push(*nn_res[0].item);
    }

    let planarity = pc1.planarity.select(Axis(0), &idx_pc1);
    let dists = dist_between_point_sets(pc1, &pts_sel_pc1, &pc2_nn_pts);
    println!("match_point_clouds took: {}", now.elapsed().as_millis());
    (planarity, dists)
}

fn dist_between_point_sets(pc1: &PointCloud, p_set1: &Vec<Item>, p_set2: &Vec<Item>) -> Array1<f64> {
    let mut dists: Array1<f64> = Array::from_elem((p_set1.len()), f64::NAN);
    for(idx, (p1, p2)) in p_set1.iter().zip(p_set2.iter()).enumerate() {
        let x1 = p1.point[0];
        let y1 = p1.point[1];
        let z1 = p1.point[2];

        let x2 = p2.point[0];
        let y2 = p2.point[1];
        let z2 = p2.point[2];

        let nx1 = pc1.normals[[p1.id, 0]];
        let ny1 = pc1.normals[[p1.id, 1]];
        let nz1 = pc1.normals[[p1.id, 2]];

        dists[[idx]] = (x2 - x1) * nx1 + (y2 - y1) * ny1 + (z2 - z1) * nz1;
    }
    dists
}

fn get_dists_median(dists: &Array1<f64>) -> f64 {
    let sorted = dists.sort_axis_by(Axis(0), |i, j| dists[[i]] > dists[[j]]);
    return if dists.len() % 2 == 0 {
        // If the length of the array is even, take the average of the two middle elements
        (dists[dists.len() / 2] + dists[(dists.len() / 2) - 1]) / 2.0
    } else {
        // If the length of the array is odd, take the middle element
        dists[dists.len() / 2]
    }
}

// https://github.com/rust-ndarray/ndarray/blob/master/examples/sort-axis.rs
pub(crate) fn reject(planarity: &Array1<f64>, dists: &Array1<f64>) {
    let now = Instant::now();

    let med = get_dists_median(dists);
    /*let keep = dists.iter().enumerate().map(|(idx, d) | {
        return if (abs(d - med) > 3 * sigmad) | (planarity_[i] < min_planarity) {
            false
        }
        else {
            true
        }
    })*/
    //let d: Array1<bool> = array![true, false, false, true, true];
    //let s = d.iter().filter(|x| **x).count();
    println!("reject took: {}", now.elapsed().as_millis());
}

#[cfg(test)]
mod corrpts_test {
    use ndarray::{array, Array, Array1};
    use crate::corrpts::get_dists_median;

    #[test]
    fn test_get_dists_median() {
        let arr1: Array1<f64> = Array::linspace(0., 1.0, 9);
        assert_eq!(get_dists_median(&arr1), 0.5);
        let arr2: Array1<f64> = Array::linspace(1., 8.0, 8);
        assert_eq!(get_dists_median(&arr2), 4.5);
    }
}