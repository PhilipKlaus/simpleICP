use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{BufWriter, Write};

use kd_tree::{ItemAndDistance, KdSlice};
use linfa_linalg::eigh::{EighInto, EigSort};
use ndarray::{Array, Array1, Array2, Axis, Ix2, s};
use ndarray_stats::CorrelationExt;

//use ndarray_stats::CorrelationExt;

pub struct PointCloud {
    points: Vec<[f64; 3]>,
    selected: Vec<bool>,
    // Store selection state for every point
    normals: Array2<f64>,
    // Store normal for every point
    planarity: Array1<f64>,
}

// Static methods
impl PointCloud {
    fn recenter_points(points: &mut Array<f64, Ix2>) {
        let mean = points.mean_axis(Axis(0)).expect("Could not calc mean");
        let mut view = points.slice_mut(s![.., ..]);
        view.zip_mut_with(&mean.view(), |a, b| *a -= b);
    }
}

#[derive(Debug, PartialEq)]
pub struct NormalRes {
    eigenvector: Array1<f64>,
    planarity: f64,
}

impl Display for NormalRes {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "NormalRes:\nEigenvector: {}\nPlanarity:{}", self.eigenvector, self.planarity)
    }
}

impl PointCloud {
    pub fn new(points: Vec<[f64; 3]>) -> PointCloud {
        let point_amount = points.len();
        PointCloud {
            points,
            selected: vec![true; point_amount], // Initially select all points,
            normals: Array::from_elem((point_amount, 3), f64::NAN),
            planarity: Array::from_elem(point_amount, f64::NAN),
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

    pub fn select_n_pts(&mut self, n: usize) {
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

            let normal = Self::normal_from_neighbors(&mut x_nn);
            self.normals[[*idx, 0]] = normal.eigenvector[[0]];
            self.normals[[*idx, 1]] = normal.eigenvector[[1]];
            self.normals[[*idx, 2]] = normal.eigenvector[[2]];
            self.planarity[[*idx]] = normal.planarity;
        }
    }

    pub fn normal_from_neighbors(neighbors: &mut Array<f64, Ix2>) -> NormalRes {
        let covariance = neighbors.t().cov(1.).unwrap();
        let eig_res = covariance.eigh_into().unwrap();
        let mut eig_sort = eig_res.sort_eig_desc();
        let eig_vals = eig_sort.0.view();

        NormalRes {
            eigenvector: eig_sort.1.slice_mut(s![.., 2]).to_owned(),
            planarity: (eig_vals[1] - eig_vals[2]) / eig_vals[0],
        }
    }
}


#[cfg(test)]
mod point_cloud_test {
    use linfa_linalg::norm::Norm;
    use ndarray::{array, Array, Ix2};

    use crate::pointcloud::{NormalRes, PointCloud};

    fn get_points() -> Array<f64, Ix2> {
        array![
            [1., 1., 1.],
            [1., 2., 1.],
            [1., 3., 1.],
            [2., 1., 2.],
            [2., 2., 2.],
            [2., 3., 2.],
            [3., 1., 3.],
            [3., 2., 3.],
            [3., 3., 3.],
        ]
    }

    #[test]
    fn recenter_points() {
        let mut points = get_points();
        PointCloud::recenter_points(&mut points);
        assert_eq!(points, array![
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

    #[test]
    fn normals_from_neighbors() {
        let mut points = get_points();
        let normal = PointCloud::normal_from_neighbors(&mut points);
        let expected = array![1., 0., -1.];
        let norm = expected.norm_l2();

        let normalized = expected / norm;

        let delta = 0.01;
        assert_float_absolute_eq!(normal.planarity, 0.5, delta);
        assert_float_absolute_eq!(normalized[[0]], 0.7, delta);
        assert_float_absolute_eq!(normalized[[1]], 0.0, delta);
        assert_float_absolute_eq!(normalized[[2]], -0.7, delta);
    }
}
