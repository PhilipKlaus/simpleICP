#[macro_use]
extern crate assert_float_eq;

use std::time::Instant;
use crate::corrpts::{reject};
use crate::pointcloud::PointCloud;

mod pointcloud;
mod corrpts;
mod permutation;
mod nearest_neighbor;

struct Parameters {
    max_overlap_distance: f64,
    correspondences: usize,
    neighbors: usize,
    max_iterations: usize,
}


fn main() {
    const FILE1: &str = "bunny1.xyz";
    const FILE2: &str = "bunny2.xyz";

    let params = Parameters {
        max_overlap_distance: 1.0,
        correspondences: 1000,
        neighbors: 10,
        max_iterations: 100,
    };

    let mut fixed = PointCloud::read_from_xyz(FILE1);
    let mut moved = PointCloud::read_from_xyz(FILE2);

    if params.max_overlap_distance > 0.0 {
        print!("Consider partial overlap of point clouds ...\n");
        fixed.select_in_range(&mut moved, params.max_overlap_distance);
        if fixed.selection_idx().len() == 0 {
            panic!(
                "Point clouds do not overlap within max_overlap_distance = {}. \
            Consider increasing the value of max_overlap_distance.",
                params.max_overlap_distance
            );
        }
        PointCloud::write_to_file(fixed.selection(), "initial_selection.xyz");
    }


    print!("Select points for correspondences in fixed point cloud ...\n");
    fixed.select_n_pts(params.correspondences);
    PointCloud::write_to_file(fixed.selection(), &*format!("select_{}_pts.xyz", params.correspondences));

    println!("Estimate normals of selected points ...\n");
    fixed.estimate_normals(params.neighbors);


    // ToDo: Needed later on for matching
    /*
    let h_old: Array2<f64> = Array::eye(4);
    let h_new: Array2<f64> = Array::from_elem((4, 4), 0.);
    let d_h: Array2<f64> = Array::from_elem((4, 4), 0.);
    */


    println!("Start iterations ...\n");
    for i in 0..params.max_iterations {
        println!("Iteration {}:", i);

        let now = Instant::now();
        let mut dists = PointCloud::cloud_to_cloud_distance(&fixed.selection(), &moved.selection());
        println!("\tdist_between_neighbors took: {}[ms]", now.elapsed().as_millis());

        reject(&fixed, &mut dists);
    }
}
