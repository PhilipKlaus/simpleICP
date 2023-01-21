use std::time::Instant;

use ndarray::{Array, Array1, ArrayBase, Axis, Ix1, OwnedRepr};

use crate::permutation::{PermuteArray, SortArray};
use crate::pointcloud::{NNRes, PointCloud};

// pc1 == fixed / pc2 == moved
pub fn dist_between_neighbors(pc1: &PointCloud, pc2: &PointCloud) -> Array1<f64> {
    let nn_res = PointCloud::knn_search(pc2, pc1, 1);
    let dists: Vec<f64> = pc1.points().outer_iter().zip(pc1.normals.outer_iter()).zip(nn_res.iter()).map(|((p1, n1), nn)| {
        let x1 = p1[[0]];
        let y1 = p1[[1]];
        let z1 = p1[[2]];

        let x2 = pc2.points()[[nn[0].idx, 0]];
        let y2 = pc2.points()[[nn[0].idx, 1]];
        let z2 = pc2.points()[[nn[0].idx, 2]];

        let nx1 = n1[[0]];
        let ny1 = n1[[1]];
        let nz1 = n1[[2]];

        (x2 - x1) * nx1 + (y2 - y1) * ny1 + (z2 - z1) * nz1
    }).collect();
    Array1::from_vec(dists)
}

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

pub(crate) fn reject(cloud: &PointCloud, dists: &mut Array1<f64>) {
    let now = Instant::now();

    let med = get_median(dists);
    let mad = get_mad(dists, med);

    /*dists.iter()
        .enumerate()
        .filter(|(idx, dist)| {
            (abs(dists_[i] - med) > 3 * sigmad) | (planarity_[i] < min_planarity)
        })
        .collect()*/

    /*
      for (int i = 0; i < dists_.size(); i++)
        {
        if ((abs(dists_[i] - med) > 3 * sigmad) | (planarity_[i] < min_planarity))
        {
          keep[i] = false;
        }
        }
    */

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
    println!("reject took: {}\n", now.elapsed().as_millis());
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