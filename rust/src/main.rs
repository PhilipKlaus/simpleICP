use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use kd_tree::{ItemAndDistance, KdSlice, KdTree};
use ndarray::prelude::*;

struct Point {
    point: [f64; 3],
    id: usize,
}

impl kd_tree::KdPoint for Point {
    type Scalar = f64;
    type Dim = typenum::U3;
    fn at(&self, k: usize) -> f64 { self.point[k] }
}

struct Parameters {
    max_overlap_distance: f32,
}

struct PointCloud {
    points: Vec<[f64; 3]>,
    selected: Vec<bool>,     // Store for every point if selected
}

impl PointCloud {
    pub fn new(points: Vec<[f64; 3]>) -> PointCloud {
        let point_amount = points.len();
        PointCloud {
            points,
            selected: vec![true; point_amount],  // Initially select all points
            //tree: KdTree::build_by_ordered_float(points),
        }
    }

    pub fn select_in_range(&self, cloud: &mut PointCloud, range: f32) {
        let sel_idx = self.get_idx_of_selected_points();
        let query_points = self.get_selected_points();

        // Get nearest neighbours
        let tree: &KdSlice<[f64; 3]> = KdSlice::sort_by_ordered_float(&mut *cloud.points);
        let mut nearests: Vec<ItemAndDistance<[f64; 3], f64>> = Vec::new();
        for query in &query_points {
            nearests.append(&mut tree.nearests(query, 1));
        }
        let mut file = File::create("selected.xyz").expect("Could not open file");
        let mut writer = BufWriter::new(file);
        for pt in nearests {
            write!(writer, "{} ", pt.item[0]).expect("Unable to write to file");
            write!(writer, "{} ", pt.item[1]).expect("Unable to write to file");
            write!(writer, "{}\n", pt.item[2]).expect("Unable to write to file");
        }
        /*
        for idx in sel_idx {

        }*/
        // Compute distances to nn
    }

    pub fn get_idx_of_selected_points(&self) -> Vec<usize> {
        self.selected.iter()
            .enumerate()
            .filter(|&(idx, state)| *state == true)
            .map(|e| e.0)
            .collect()
    }

    pub fn get_selected_points(&self) -> Vec<[f64; 3]> {
        self.get_idx_of_selected_points().into_iter().map(|idx| self.points[idx]).collect()
    }
}

// ToDo: Return Option
fn read_xyz_file(path: String) -> Vec<[f64; 3]> {
    let file = File::open(path).expect("Unable to open file");
    let reader = BufReader::new(file);
    let res: Vec<[f64; 3]> = reader.lines().into_iter().map(|l| {
        let line = l.expect("Could not read line");
        let mut parts = line.split_whitespace();
        let x: f64 = parts.next().expect("Unable to parse x coordinate")
            .parse().expect("Unable to parse x coordinate");
        let y: f64 = parts.next().expect("Unable to parse y coordinate")
            .parse().expect("Unable to parse y coordinate");
        let z: f64 = parts.next().expect("Unable to parse z coordinate")
            .parse().expect("Unable to parse z coordinate");
        [x, y, z]
    }).collect();

    res
}

fn main() {
    let params = Parameters {
        max_overlap_distance: 1.0
    };

    let p1 = read_xyz_file("bunny1.xyz".to_string());
    let p2 = read_xyz_file("bunny2.xyz".to_string());
    let mut fixed = PointCloud::new(p1);
    let mut moved = PointCloud::new(p2);

    if params.max_overlap_distance > 0.0
    {
        print!("Consider partial overlap of point clouds ...\n");
        fixed.select_in_range(&mut moved, params.max_overlap_distance);
        /*
        pc_fix.SelectInRange(pc_mov.X(), max_overlap_distance);
    if (pc_fix.GetIdxOfSelectedPts().size() == 0)
    {
      char buff[200];
      snprintf(buff, sizeof(buff),
               "Point clouds do not overlap within max_overlap_distance = %.5f. "
               "Consider increasing the value of max_overlap_distance.\n",
               max_overlap_distance);
      std::string error_msg{buff};
      throw std::runtime_error(error_msg);
    }
        */
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
