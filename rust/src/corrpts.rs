use std::cmp::Ordering;
use std::ptr::copy_nonoverlapping;
use std::time::Instant;

use ndarray::{Array, Array1, Axis};
use ndarray::{Data, RemoveAxis, Zip};
use ndarray::prelude::*;
use rawpointer::PointerExt;
use crate::permutation::SortArray;

use crate::pointcloud::{NNRes, PointCloud};

static mut called: u64 = 0;

// pc1 == fixed / pc2 == moved
pub fn match_point_clouds(pc1: &PointCloud, pc2: &PointCloud) -> (Array1<f64>, Array1<f64>) {
    let now = Instant::now();

    let pts_sel_pc1 = pc1.get_selected_points();
    let mut pts_sel_pc2 = pc2.get_selected_points();

    let idx_pc1 = pc1.get_idx_of_selected_points();

    let nn = PointCloud::knn_search(&mut pts_sel_pc2, &pts_sel_pc1, 1);

    let planarity = pc1.planarity.select(Axis(0), &idx_pc1);
    let dists = dist_between_neighbors(pc1, &idx_pc1, pc2, &nn);
    println!("match_point_clouds took: {}", now.elapsed().as_millis());
    (planarity, dists)
}

fn dist_between_neighbors(pc1: &PointCloud, idx_pc1: &Vec<usize>, pc2: &PointCloud, neighbors: &Vec<Vec<NNRes>>) -> Array1<f64> {
    let mut dists: Array1<f64> = Array::from_elem(idx_pc1.len(), f64::NAN);
    for (idx, (p1, nn)) in idx_pc1.iter().zip(neighbors.iter()).enumerate() {
        let x1 = pc1.points[[*p1, 0]];
        let y1 = pc1.points[[*p1, 1]];
        let z1 = pc1.points[[*p1, 2]];

        let x2 = pc2.points[[nn[0].idx, 0]];
        let y2 = pc2.points[[nn[0].idx, 1]];
        let z2 = pc2.points[[nn[0].idx, 2]];

        let nx1 = pc1.normals[[*p1, 0]];
        let ny1 = pc1.normals[[*p1, 1]];
        let nz1 = pc1.normals[[*p1, 2]];

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
    };
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