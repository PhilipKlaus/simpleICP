use std::fmt::format;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use kd_tree::{ItemAndDistance, KdSlice, KdTree};
use ndarray::prelude::*;

struct Parameters {
    max_overlap_distance: f64,
    correspondences: usize,
    neighbors: usize,
}

struct PointCloud {
    points: Vec<[f64; 3]>,
    selected: Vec<bool>,
    // Store selection state for every point
    normals: Array2<f64>, // Store normal for every point
}

// Static methods
impl PointCloud {
    fn recenter_points(points: &mut Array<f64, Ix2>) {
        let mean = points.mean_axis(Axis(0)).expect("Could not calc mean");
        let mut view = points.slice_mut(s![.., ..]);
        view.zip_mut_with(&mean.view(), |a, b| *a -= b);
    }
}

impl PointCloud {
    pub fn new(points: Vec<[f64; 3]>) -> PointCloud {
        let point_amount = points.len();
        PointCloud {
            points,
            selected: vec![true; point_amount], // Initially select all points,
            normals: Array::from_elem((point_amount, 3), f64::NAN),
            //tree: KdTree::build_by_ordered_float(points),
        }
    }

    pub fn knn_search<'a>(
        cloud_ref: &'a mut Vec<[f64; 3]>,
        cloud_query: &'a Vec<[f64; 3]>,
        k: usize,
    ) -> Vec<Vec<ItemAndDistance<'a, [f64; 3], f64>>> {
        let tree: &KdSlice<[f64; 3]> = KdSlice::sort_by_ordered_float(cloud_ref);
        let mut nn: Vec<Vec<ItemAndDistance<[f64; 3], f64>>> = Vec::new();
        for query in cloud_query {
            nn.push(tree.nearests(query, k));
        }
        nn
    }

    pub fn select_in_range(&mut self, cloud: &mut PointCloud, max_range: f64) {
        let sel_idx = self.get_idx_of_selected_points();
        let query_points = self.get_selected_points();

        // Get nearest neighbours
        let nn = PointCloud::knn_search(&mut cloud.points, &query_points, 1);

        // ToDo: possible impr. compare to squared dist -> than no sqrt necessary
        for i in 0..sel_idx.len() {
            if f64::sqrt(nn[i][0].squared_distance) > max_range {
                self.selected[sel_idx[i]] = false;
            }
        }
    }

    pub fn export_selected_points(&self, name: &str) {
        let mut file = File::create(name).expect("Could not open file");
        let mut writer = BufWriter::new(file);
        for pt in self.get_selected_points() {
            write!(writer, "{} ", pt[0]).expect("Unable to write to file");
            write!(writer, "{} ", pt[1]).expect("Unable to write to file");
            write!(writer, "{}\n", pt[2]).expect("Unable to write to file");
        }
    }

    pub(crate) fn select_n_pts(&mut self, n: usize) {
        let sel_idx = self.get_idx_of_selected_points();
        if n < sel_idx.len() {
            self.selected = vec![false; self.points.len()];
            let dist_idx = Array::<f64, _>::linspace(0., (sel_idx.len() - 1) as f64, n);

            for (i, idx) in dist_idx.iter().enumerate() {
                self.selected[sel_idx[idx.floor() as usize]] = true;
            }
        }
    }

    pub fn get_idx_of_selected_points(&self) -> Vec<usize> {
        self.selected
            .iter()
            .enumerate()
            .filter(|&(idx, state)| *state == true)
            .map(|e| e.0)
            .collect()
    }

    pub fn get_selected_points(&self) -> Vec<[f64; 3]> {
        self.get_idx_of_selected_points()
            .into_iter()
            .map(|idx| self.points[idx])
            .collect()
    }

    pub fn estimate_normals(&mut self, neighbors: usize) {
        let sel_idx = self.get_idx_of_selected_points();
        let query_points = self.get_selected_points();

        let point_amount = self.points.len();
        let mut nn = PointCloud::knn_search(&mut self.points, &query_points, neighbors);

        self.normals = Array::from_elem((point_amount, 3), f64::NAN);

        for (i, idx) in sel_idx.iter().enumerate() {
            let mut x_nn: Array<f64, Ix2> = Array::zeros((neighbors, 3));

            for j in 0..neighbors {
                x_nn[[j, 0]] = nn[i][j].item[0];
                x_nn[[j, 1]] = nn[i][j].item[1];
                x_nn[[j, 2]] = nn[i][j].item[2];
            }

            PointCloud::recenter_points(&mut x_nn);
        }
    }
}

// ToDo: Return Option
fn read_xyz_file(path: String) -> Vec<[f64; 3]> {
    let file = File::open(path).expect("Unable to open file");
    let reader = BufReader::new(file);
    let res: Vec<[f64; 3]> = reader
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
        .collect();

    res
}

fn main() {
    let params = Parameters {
        max_overlap_distance: 1.0,
        correspondences: 1000,
        neighbors: 10,
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
        fixed.export_selected_points("initial_selection.xyz");
    }

    print!("Select points for correspondences in fixed point cloud ...\n");
    fixed.select_n_pts(params.correspondences);
    fixed.export_selected_points(&*format!("select_{}_pts.xyz", params.correspondences));

    println!("Estimate normals of selected points ...\n");
    fixed.estimate_normals(params.neighbors);
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

#[cfg(test)]
mod point_cloud_test {
    use ndarray::{array, Array, Ix2};

    use crate::PointCloud;

    #[test]
    fn recenter_points() {
        let mut points_to_center = array![
            [1., 1., 1.],
            [1., 2., 1.],
            [1., 3., 1.],
            [2., 1., 2.],
            [2., 2., 2.],
            [2., 3., 2.],
            [3., 1., 3.],
            [3., 2., 3.],
            [3., 3., 3.],
        ];
        PointCloud::recenter_points(&mut points_to_center);
        assert_eq!(points_to_center, array![
            [-1., -1., -1.],
            [-1., 0., -1.],
            [-1., 1., -1.],
            [0., -1., 0.],
            [0., 0., 0.],
            [0., 1., 0.],
            [1., -1., 1.],
            [1., 0., 1.],
            [1., 1., 1.],
        ]);
    }
}
