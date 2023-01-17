use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use linfa_linalg::eigh::{EighInto, EigSort};
use ndarray::{Array, Array1, Array2, ArrayView, Axis, Ix1, Ix2, s, stack};
use ndarray_stats::CorrelationExt;
use crate::point_selection::{PointCloudView, PointSelection};

#[derive(Debug, PartialEq)]
pub struct NormalRes {
    eigenvector: Array1<f64>,
    planarity: f64,
}

pub struct NNRes {
    pub(crate) distance: f64,
    pub(crate) idx: usize,
}

impl From<(f64, usize)> for NNRes {
    fn from(value: (f64, usize)) -> Self {
        NNRes { distance: value.0, idx: value.1 }
    }
}

impl Display for NormalRes {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "NormalRes:\nEigenvector: {}\nPlanarity:{}", self.eigenvector, self.planarity)
    }
}



pub struct PointCloud {
    pub points: Array2<f64>,
    // Store selection state for every point
    pub normals: Array2<f64>,
    // Store normal for every point
    pub planarity: Array1<f64>,

    pub selected: Array1<bool>,
    selection: Option<PointSelection>,
}

impl PointCloudView for PointCloud {
    fn x(&self) -> ArrayView<'_, f64, Ix2> {
        self.points.view()
    }
}


//###############################
//# 'Static' PointCloud methods #
//###############################
impl PointCloud {
    // ToDo: return Result<>
    pub fn read_from_xyz(path: &str) -> PointCloud {
        let file = File::open(path).expect("Could not read pointcloud from file");
        let reader = BufReader::new(file);
        let point_data = reader
            .lines()
            .into_iter()
            .map(|l| {
                let line = l.expect("Could not read line");
                let xyz: Vec<f64> = line.split_whitespace().map(|part| {
                    let coord: f64 = part.parse().expect("Unable to parse coordinate");
                    coord
                }).collect();
                xyz
            })
            .flatten()
            .collect();
        PointCloud::new(point_data)
    }

    pub fn write_to_file(cloud: &dyn PointCloudView, name: &str) {
        let file = File::create(name).expect("Could not open file");
        let mut writer = BufWriter::new(file);
        for pt in cloud.x().outer_iter() {
            write!(writer, "{} ", pt[[0]]).expect("Unable to write to file");
            write!(writer, "{} ", pt[[1]]).expect("Unable to write to file");
            write!(writer, "{}\n", pt[[2]]).expect("Unable to write to file");
        }
    }

    pub fn knn_search(
        reference: &dyn PointCloudView,
        query: &dyn PointCloudView,
        k: usize,
    ) -> Vec<Vec<NNRes>> {
        let mut kdtree = KdTree::new(3);

        for (idx, p) in reference.x().outer_iter().enumerate() {
            kdtree.add([p[[0]], p[[1]], p[[2]]], idx).expect("Could not add point to kdtree");
        }

        let mut nn: Vec<Vec<NNRes>> = Vec::new();
        for q in query.x().outer_iter() {
            nn.push(
                kdtree.nearest(&[q[[0]], q[[1]], q[[2]]], k, &squared_euclidean)
                    .expect("Could not fetch nn for point")
                    .iter()
                    .map(|entry| NNRes::from((entry.0, *entry.1)))
                    .collect()
            );
        }
        nn
    }

    // Alternative implementation using ArrayViews
    #[allow(dead_code)]
    pub fn knn_search_e<'a>(
        cloud_ref: &Vec<ArrayView<f64, Ix1>>,
        cloud_query: &Vec<ArrayView<f64, Ix1>>,
        k: usize,
    ) -> Vec<Vec<NNRes>> {
        let mut kdtree = KdTree::new(3);

        for (idx, p) in cloud_ref.iter().enumerate() {
            kdtree.add([p[[0]], p[[1]], p[[2]]], idx).expect("Could not add point to kdtree");
        }

        let mut nn: Vec<Vec<NNRes>> = Vec::new();
        for query in cloud_query.iter() {
            nn.push(
                kdtree.nearest(&[query[[0]], query[[1]], query[[2]]], k, &squared_euclidean)
                    .expect("Could not fetch nn for point")
                    .iter()
                    .map(|entry| NNRes::from((entry.0, *entry.1)))
                    .collect()
            );
        }
        nn
    }
}

//###############################
//# 'Getters' for PointCloud    #
//###############################
impl PointCloud {
    pub fn point_amount(&self) -> usize {
        self.points.shape()[0]
    }
}

impl PointCloud {
    pub fn new(points: Vec<f64>) -> PointCloud {
        let point_amount = points.len() / 3;
        PointCloud {
            points: Array::from_shape_vec((point_amount, 3), points)
                .expect("Could not create ndarray from points"),
            selected: Array1::from_elem(point_amount, true), // Initially select all points,
            normals: Array::from_elem((point_amount, 3), f64::NAN),
            planarity: Array::from_elem(point_amount, f64::NAN),
            selection: None,
        }
    }


    pub fn select_in_range(&mut self, cloud: &mut PointCloud, max_range: f64) {
        let sel_idx = self.get_idx_of_selected_points();
        let query = self.get_selected_points();

        // Get nearest neighbours
        let nn = PointCloud::knn_search(cloud, query, 1);

        // ToDo: possible impr. compare to squared dist -> than no sqrt necessary
        for i in 0..sel_idx.len() {
            if f64::sqrt(nn[i][0].distance) > max_range {
                self.selected[sel_idx[i]] = false;
            }
        }

        self.selection = Option::from(PointSelection::select_from_point_cloud(self, &self.get_idx_of_selected_points()));
    }

    pub fn select_n_pts(&mut self, n: usize) {
        let sel_idx = self.get_idx_of_selected_points();
        if n < sel_idx.len() {
            self.selected.fill(false);
            let dist_idx = Array::<f64, _>::linspace(0., (sel_idx.len() - 1) as f64, n);

            for idx in dist_idx.iter() {
                self.selected[sel_idx[idx.floor() as usize]] = true;
            }

            self.selection = Option::from(PointSelection::select_from_point_cloud(self, &self.get_idx_of_selected_points()));
        }
    }

    pub fn get_idx_of_selected_points(&self) -> Vec<usize> {
        self.selected
            .iter()
            .enumerate()
            .filter(|&(_, state)| *state == true)
            .map(|e| e.0)
            .collect()
    }

    pub fn get_selected_points(&self) -> &dyn PointCloudView {
        match self.selection {
            Some(ref x) => x,
            None => self
        }
    }

    pub fn estimate_normals(&mut self, neighbors: usize) {
        let sel_idx = self.get_idx_of_selected_points();
        let query_points = self.get_selected_points();

        let point_amount = self.points.len();
        let nn = PointCloud::knn_search(self, query_points, neighbors);

        self.normals = Array::from_elem((point_amount, 3), f64::NAN);

        for (i, idx) in sel_idx.iter().enumerate() {
            let mut x_nn: Array<f64, Ix2> = Array::zeros((neighbors, 3));

            for j in 0..neighbors {
                x_nn[[j, 0]] = self.points[[nn[i][j].idx, 0]];
                x_nn[[j, 1]] = self.points[[nn[i][j].idx, 1]];
                x_nn[[j, 2]] = self.points[[nn[i][j].idx, 2]];
            }

            let normal = Self::normal_from_neighbors(&mut x_nn);
            self.normals[[*idx, 0]] = normal.eigenvector[[0]];
            self.normals[[*idx, 1]] = normal.eigenvector[[1]];
            self.normals[[*idx, 2]] = normal.eigenvector[[2]];
            self.planarity[[*idx]] = normal.planarity;
        }
    }

    fn normal_from_neighbors(neighbors: &mut Array<f64, Ix2>) -> NormalRes {
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
