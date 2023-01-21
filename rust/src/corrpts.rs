use std::time::Instant;

use ndarray::{Array, Array1, Axis, Ix1};

use crate::permutation::{PermuteArray, SortArray};
use crate::point_selection::PointCloudView;
use crate::pointcloud::{NNRes, PointCloud};

// pc1 == fixed / pc2 == moved
pub fn match_point_clouds(pc1: &PointCloud, pc2: &PointCloud) -> (Array1<f64>, Array1<f64>) {
    let now = Instant::now();

    let pts_sel_pc1 = pc1.get_selected_points();
    let pts_sel_pc2 = pc2.get_selected_points();

    let idx_pc1 = pc1.get_idx_of_selected_points();

    let nn = PointCloud::knn_search(pts_sel_pc2, pts_sel_pc1, 1);

    let planarity = pc1.planarity.select(Axis(0), &idx_pc1);
    let dists = dist_between_neighbors(pc1, &idx_pc1, pc2, &nn);
    println!("match_point_clouds took: {}", now.elapsed().as_millis());
    (planarity, dists)
}

fn dist_between_neighbors(pc1: &PointCloud, idx_pc1: &Vec<usize>, pc2: &PointCloud, neighbors: &Vec<Vec<NNRes>>) -> Array1<f64> {
    let mut dists: Array1<f64> = Array::from_elem(idx_pc1.len(), f64::NAN);
    for (idx, (p1, nn)) in idx_pc1.iter().zip(neighbors.iter()).enumerate() {
        let x1 = pc1.x()[[*p1, 0]];
        let y1 = pc1.x()[[*p1, 1]];
        let z1 = pc1.x()[[*p1, 2]];

        let x2 = pc2.x()[[nn[0].idx, 0]];
        let y2 = pc2.x()[[nn[0].idx, 1]];
        let z2 = pc2.x()[[nn[0].idx, 2]];

        let nx1 = pc1.normals[[*p1, 0]];
        let ny1 = pc1.normals[[*p1, 1]];
        let nz1 = pc1.normals[[*p1, 2]];

        dists[[idx]] = (x2 - x1) * nx1 + (y2 - y1) * ny1 + (z2 - z1) * nz1;
    }
    dists
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

pub(crate) fn reject(planarity: &Array1<f64>, dists: &Array1<f64>) {
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