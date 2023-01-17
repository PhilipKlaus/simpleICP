use ndarray::{Array2, ArrayView, Axis, Ix2};

pub trait PointCloudView {
    fn x(&self) -> ArrayView<'_, f64, Ix2>;
}

pub struct PointSelection {
    points: Array2<f64>,
}

impl PointSelection {
    pub fn new(points: Array2<f64>) -> PointSelection {
        PointSelection { points }
    }

    pub fn select_from_point_cloud(cloud: &dyn PointCloudView, idx: &Vec<usize>) -> PointSelection {
        PointSelection { points: cloud.x().select(Axis(0), &idx) }
    }
}

impl PointCloudView for PointSelection {
    fn x(&self) -> ArrayView<'_, f64, Ix2> {
        self.points.view()
    }
}