use std::fmt::{Display, Formatter};
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use ndarray::Array1;
use crate::pointcloud::PointCloud;

#[derive(Debug, PartialEq)]
pub struct NormalRes {
    pub eigenvector: Array1<f64>,
    pub planarity: f64,
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

pub fn knn_search(
    reference: &PointCloud,
    query: &PointCloud,
    k: usize,
) -> Vec<Vec<NNRes>> {
    let mut kdtree = KdTree::new(3);

    for (idx, p) in reference.points().outer_iter().enumerate() {
        kdtree.add([p[[0]], p[[1]], p[[2]]], idx).expect("Could not add point to kdtree");
    }

    let mut nn: Vec<Vec<NNRes>> = Vec::new();
    for q in query.points().outer_iter() {
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