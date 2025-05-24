#include "camera.h"

#include <opencv2/core/core.hpp>

namespace mango {

Camera::Camera(int id, const std::string& name)
    : id_(id), name_(name), fx(0.0), fy(0.0), cx(0.0), cy(0.0), k1(0.0), k2(0.0)
{

}

Camera::~Camera(){}

void Camera::loadConfig(const std::string& file)
{
    cv::FileStorage fs(file, cv::FileStorage::READ);
    
    if(!fs.isOpened())
    {
        return;
    }
    // 从配置文件中读取相机的内参和畸变参数
    // fx, fy, cx, cy = 0.0;
    // k1, k2 = 0.0;
    cv::FileNode n = fs["camera"]["intrinsics"];
    fx = static_cast<double>(n["fx"]);
    fy = static_cast<double>(n["fy"]);
    cx = static_cast<double>(n["cx"]);
    cy = static_cast<double>(n["cy"]);

    n = fs["camera"]["distort"];
    k1 = static_cast<double>(n["k1"]);
    k2 = static_cast<double>(n["k2"]);
}

}