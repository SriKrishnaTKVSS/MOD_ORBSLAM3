#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Include this if you use STL containers (std::vector, std::string)
#include <System.h>  // Include your C++ headers

namespace py = pybind11;

PYBIND11_MODULE(ORB_SLAM3_python, m) {
    // Create a binding for the ORB_SLAM3::System class
    py::class_<ORB_SLAM3::System>(m, "System")
        .def(py::init<const std::string&, const std::string&, ORB_SLAM3::System::eSensor, bool>(),
            py::arg("vocab_file"), py::arg("settings_file"), py::arg("sensor"), py::arg("use_viewer"))
        .def("TrackMonocular", &ORB_SLAM3::System::TrackMonocular)
        .def("Shutdown", &ORB_SLAM3::System::Shutdown)
        .def("SaveTrajectoryTUM", &ORB_SLAM3::System::SaveTrajectoryTUM)
        .def("SaveKeyFrameTrajectoryTUM", &ORB_SLAM3::System::SaveKeyFrameTrajectoryTUM);

    // Expose enums, other classes, or free functions if necessary
    py::enum_<ORB_SLAM3::System::eSensor>(m, "eSensor")
        .value("MONOCULAR", ORB_SLAM3::System::MONOCULAR)
        .value("STEREO", ORB_SLAM3::System::STEREO)
        .value("RGBD", ORB_SLAM3::System::RGBD);

    // You can add more bindings for other classes or functions as needed
}
