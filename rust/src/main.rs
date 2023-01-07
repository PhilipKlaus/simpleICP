#[macro_use]
extern crate assert_float_eq;

use std::fmt::format;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

use kd_tree::{ItemAndDistance, KdSlice};
use kdtree::KdTree;
use kdtree::ErrorKind;
use kdtree::distance::squared_euclidean;use ndarray::prelude::*;
use ordered_float::OrderedFloat;

use crate::corrpts::{match_point_clouds, reject};
use crate::pointcloud::{Item, PointCloud};

mod pointcloud;
mod corrpts;

struct Parameters {
    max_overlap_distance: f64,
    correspondences: usize,
    neighbors: usize,
    max_iterations: usize,
}

// ToDo: Return Option
fn read_xyz_file(path: String) -> Vec<Item> {
    let file = File::open(path).expect("Unable to open file");
    let reader = BufReader::new(file);
    reader
        .lines()
        .into_iter()
        .enumerate()
        .map(|(idx, l)| {
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
            Item {
                point: [x, y, z],
                id: idx,
            }
        })
        .collect()
}

fn read_xyz_file_simple(path: String) -> Vec<[f64; 3]> {
    let file = File::open(path).expect("Unable to open file");
    let reader = BufReader::new(file);
    reader
        .lines()
        .into_iter()
        .enumerate()
        .map(|(idx, l)| {
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
        .collect()
}

fn main() {

    let params = Parameters {
        max_overlap_distance: 1.0,
        correspondences: 1000,
        neighbors: 10,
        max_iterations: 100,
    };

    let mut p1 = read_xyz_file("bunny1.xyz".to_string());
    let mut p2 = read_xyz_file("bunny2.xyz".to_string());
    /*
    // kd-tree crate: Using Vector of Item
    let mut p1 = read_xyz_file("bunny1.xyz".to_string());
    let mut p2 = read_xyz_file("bunny2.xyz".to_string());
    let now = Instant::now();
    let tree = kd_tree::KdSlice3::sort_by_key(&mut p1, |item, k| OrderedFloat(item.point[k]));
    let mut nn: Vec<Vec<ItemAndDistance<Item, f64>>> = Vec::new();
    for query in p2 {
        nn.push(tree.nearests(&query, 10));
    }
    println!("knn_search took: {}", now.elapsed().as_millis());

    // kd-tree crate: Using Vector of [f64;3]
    let mut p1 = read_xyz_file_simple("bunny1.xyz".to_string());
    let mut p2 = read_xyz_file_simple("bunny2.xyz".to_string());
    let now = Instant::now();
    let tree: &KdSlice<[f64; 3]> = KdSlice::sort_by_ordered_float(&mut p1);
    let mut nn: Vec<Vec<ItemAndDistance<[f64; 3], f64>>> = Vec::new();
    for query in p2 {
        nn.push(tree.nearests(&query, 10));
    }
    println!("knn_search took: {}", now.elapsed().as_millis());

    // Using kdtree crate
    let dimensions = 3;
    let mut p1 = read_xyz_file_simple("bunny1.xyz".to_string());
    let mut p2 = read_xyz_file_simple("bunny2.xyz".to_string());
    let now = Instant::now();

    let mut kdtree= KdTree::new(dimensions);

    for (idx, p) in p1.iter().enumerate() {
        kdtree.add(p, idx);
    }

    let mut nn: Vec<Vec<(f64, &usize)>> =Vec::new();
    for (idx, query) in p2.iter().enumerate() {
        nn.push(kdtree.nearest(query, 10, &squared_euclidean).unwrap());
    }
    println!("knn_search took: {}", now.elapsed().as_millis());
    */

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
        fixed.export_selected_points("initial_selection.xyz");
    }

    print!("Select points for correspondences in fixed point cloud ...\n");
    fixed.select_n_pts(params.correspondences);
    fixed.export_selected_points(&*format!("select_{}_pts.xyz", params.correspondences));

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

    /*

    let a = array![
                [1.,2.,3.],
                [4.,5.,6.],
                [7.,8.,9.],
                [10.,11.,12.],
                [13.,14.,15.],
                [16.,17.,18.],
                [19.,20.,21.],
            ];

    println!("{:?}", a);

    let coords:Vec<[f64;3]>= a.into_raw_vec().chunks(3).map(|chunk| [chunk[0], chunk[1], chunk[2]]).collect();
    let kdtree: KdTree<[f64; 3]> = KdTree::build_by_ordered_float(coords);
    */
}
