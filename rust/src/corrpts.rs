use std::borrow::Borrow;
use std::iter::Filter;
use std::num::FpCategory::Nan;
use std::time::Instant;

use itertools::izip;
use ndarray::{Array, Array1, Axis, Ix1};

use crate::permutation::{PermuteArray, SortArray};
use crate::pointcloud::{CloudToCloudDist, PointCloud};

fn get_median(data: &Array1<f64>) -> f64 {
    let data_copy = data.clone();
    let perm = data_copy.sort_axis_by(Axis(0), |i, j| data_copy[[i]] > data_copy[[j]]);
    let sorted = data_copy.permute_axis(Axis(0), &perm);
    let len = data.len();
    return if len % 2 == 0 {
        // If the length of the array is even, take the average of the two middle elements
        (sorted[len / 2] + sorted[(len / 2) - 1]) / 2.0
    } else {
        // If the length of the array is odd, take the middle element
        sorted[len / 2]
    };
}

fn get_mad(data: &Array1<f64>, median: f64) -> f64 {
    let dmed: Array<f64, Ix1> = data.map(|x| f64::abs(x - median));
    get_median(&dmed)
}

// Returns valid point indices indices
pub(crate) fn reject(cloud: &PointCloud, dist: &mut CloudToCloudDist, min_planarity: usize) -> Vec<usize> {

    let med = get_median(dist.dist.borrow());
    let mad = get_mad(dist.dist.borrow(), med);
    let sigmad = 1.4826 * mad;

    assert_eq!(cloud.point_amount(), dist.nn.len());

    let keep: Vec<usize> = izip!(cloud.planarity().iter(), dist.dist.iter())
        .enumerate()
        .filter(|(idx, (p, dist))| {
            (f64::abs(**dist - med) <= 3.0 * sigmad)//(f64::abs(**dist - med) <= 3.0 * sigmad) && (**p >= min_planarity as f64)
        })
        .map(|(idx, _)| {
            idx
        })
        .collect();

    keep
}

#[cfg(test)]
mod corrpts_test {
    use ndarray::{array, Array, Array1};

    use crate::corrpts::get_median;

    #[test]
    fn test_get_dists_median() {
        let arr1: Array1<f64> = Array::linspace(0., 1.0, 9);
        assert_eq!(get_median(&arr1), 0.5);
        let arr2: Array1<f64> = Array::linspace(1., 8.0, 8);
        assert_eq!(get_median(&arr2), 4.5);
        let arr3: Array1<f64> = array![3.0, 0.0, 1.0]; // Test that values are sorted beforehand
        assert_eq!(get_median(&arr3), 1.0);
    }
}