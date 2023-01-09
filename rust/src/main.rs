#[macro_use]
extern crate assert_float_eq;

use std::fs::File;
use std::io::{BufRead, BufReader};

use ndarray::prelude::*;

use crate::corrpts::{match_point_clouds, reject};
use crate::pointcloud::PointCloud;

mod pointcloud;
mod corrpts;
mod permutation;

struct Parameters {
    max_overlap_distance: f64,
    correspondences: usize,
    neighbors: usize,
    max_iterations: usize,
}

// ToDo: Return Option
fn read_xyz_file(path: String) -> Vec<f64> {
    let file = File::open(path).expect("Unable to open file");
    let reader = BufReader::new(file);
    reader
        .lines()
        .into_iter()
        .map(|l| {
            let line = l.expect("Could not read line");
            let mut parts = line.split_whitespace();
            let x: f64 = parts
                .next()
                .expect("Unable to parse x coordinate")
                .parse()
                .expect("Unable to parse x coordinate");
            let y: f64 = parts
                .next()
                .expect("Unable to parse y coordinate")
                .parse()
                .expect("Unable to parse y coordinate");
            let z: f64 = parts
                .next()
                .expect("Unable to parse z coordinate")
                .parse()
                .expect("Unable to parse z coordinate");
            [x, y, z]
        })
        .flatten()
        .collect()
}

fn main() {
    let params = Parameters {
        max_overlap_distance: 1.0,
        correspondences: 1000,
        neighbors: 10,
        max_iterations: 100,
    };

    let p1 = read_xyz_file("bunny1.xyz".to_string());
    let p2 = read_xyz_file("bunny2.xyz".to_string());

    let mut fixed = PointCloud::new(p1);
    let mut moved = PointCloud::new(p2);

    if params.max_overlap_distance > 0.0 {
        print!("Consider partial overlap of point clouds ...\n");
        fixed.select_in_range(&mut moved, params.max_overlap_distance);
        if fixed.get_idx_of_selected_points().len() == 0 {
            panic!(
                "Point clouds do not overlap within max_overlap_distance = {}. \
            Consider increasing the value of max_overlap_distance.",
                params.max_overlap_distance
            );
        }
        PointCloud::write_cloud_to_file(fixed.get_selected_points(), "initial_selection.xyz");
    }

    print!("Select points for correspondences in fixed point cloud ...\n");
    fixed.select_n_pts(params.correspondences);
    PointCloud::write_cloud_to_file(fixed.get_selected_points(), &*format!("select_{}_pts.xyz", params.correspondences));

    println!("Estimate normals of selected points ...\n");
    fixed.estimate_normals(params.neighbors);

    let h_old: Array2<f64> = Array::eye(4);
    let h_new: Array2<f64> = Array::from_elem((4, 4), 0.);
    let d_h: Array2<f64> = Array::from_elem((4, 4), 0.);

    println!("Start iterations ...\n");
    for i in 0..params.max_iterations {
        let match_res = match_point_clouds(&fixed, &moved); // planarity, dists
        reject(&match_res.0, &match_res.1);
    }
}
