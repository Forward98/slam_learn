#include <fstream>
#include <string>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "camera.h"
#include "camera_mgr.h"
#include "pose.hpp"
#include "point.hpp"
#include "util.h"

static int camera_id = 0;
std::string dataFolder = "E:/VScode_project/slam_learn/data/"; // 绝对路径
std::string confFolder = "E:/VScode_project/slam_learn/conf/"; // 绝对路径

/**
 * 世界坐标系立方体坐标点，从配置读取
*/
cv::Mat loadCube(const std::string& file)
{
    cv::Mat cube;
    cv::FileStorage fs(file, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        return cube;
    }

    fs["cube"] >> cube;
    return cube; 
}

/**
 * 在图像上绘制立方体
*/
cv::Mat drawCube(const cv::Mat& img, const mango::Pose3D& pose)
{
    // cube 是一个 cv::Mat 类型的矩阵，包含 8 个顶点的 3D 坐标（每个顶点有 x、y、z 三个坐标值）。
    cv::Mat cube = loadCube(confFolder + "camera.yaml");
    int r = cube.rows, c = cube.cols;
    // 确保立方体有 8 个顶点，每个顶点有 3 个坐标值。
    assert(r == 8 && c == 3);

    // 创建一个与输入图像大小相同的彩色图像 cubeImg。
    cv::Mat cubeImg(img.rows, img.cols, CV_8UC3);
    // 如果输入图像是灰度图（单通道），则将其转换为彩色图（三通道）。
    if(img.channels() == 1)
    {
        std::vector<cv::Mat> channels(3, img);
        cv::merge(channels, cubeImg);
    }
    else
    {
        // 如果输入图像是彩色图，则直接复制到 cubeImg 中。
        img.copyTo(cubeImg);
    }

    std::vector<cv::Point> points;
    // 遍历立方体的每个顶点。
    for(int i = 0; i < r; i++)
    {
        double x = cube.at<double>(i, 0);
        double y = cube.at<double>(i, 1);
        double z = cube.at<double>(i, 2);
        // 使用 project 函数将 3D 顶点投影到 2D 图像平面上，得到 2D 坐标点p。
        mango::Point2D p = project(mango::Point3D(x, y, z), mango::CameraMgr::getInstance().getCameraById(camera_id), pose);
        points.push_back(cv::Point(p.x, p.y));
        // 在 cubeImg 上绘制每个顶点，使用红色圆点标记。
        cv::circle(cubeImg, cv::Point(p.x, p.y), 2, cv::Scalar(0,0,255), -1);
    }
    // 使用 line 函数绘制立方体的 12 条边。
    line(cubeImg, points[0], points[1], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[1], points[2], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[2], points[3], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[3], points[0], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[4], points[5], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[5], points[6], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[6], points[7], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[7], points[4], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[0], points[4], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[1], points[5], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[2], points[6], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[3], points[7], cv::Scalar(0,0,255), 2);

    return cubeImg;
}

void run(const cv::Mat& img, const mango::Pose3D& pose)
{
    // w 和 h 分别存储输入图像的宽度和高度。
    int w = img.cols, h = img.rows;
    // 对图像进行畸变校正
    cv::Mat undistort_img = mango::undistortImage(img, mango::CameraMgr::getInstance().getCameraById(camera_id));
    // 在校正后的图像上绘制一个 3D 立方体, 绘制了立方体的图像存储在 drawcube_img 中。
    cv::Mat drawcube_img = drawCube(undistort_img, pose);

    // 创建一个包含原始图像和绘制了立方体的图像的向量 imgs。
    // 调用 mergeImage 函数将这两个图像合并成一个图像。
    // 合并后的图像存储在 merge_img 中。
    std::vector<cv::Mat> imgs{img, drawcube_img};
    cv::Mat merge_img = mango::mergeImage(imgs, w, h);

    cv::imshow("result", merge_img);
    cv::waitKey(25);
}


int main(int argc, char** argv)
{
    // 初始化camera
    // 程序的主函数，负责初始化相机、读取位姿文件和图像文件，并逐帧处理图像。
    static mango::CameraPtr camera_ = std::shared_ptr<mango::Camera>(new mango::Camera(++camera_id, "default camera"));
    camera_->loadConfig(confFolder + "camera.yaml");
    mango::CameraMgr::getInstance().addCamera(camera_);

    // 每帧相机位姿
    std::vector<mango::Pose3D> poses;
    try
    {
        std::string posesFile = dataFolder + "poses.txt";
        std::ifstream ifPoses(posesFile.c_str());
        if(!ifPoses.is_open())
        {
            std::cerr << "找不到文件：" << posesFile << std::endl;
            return -1;
        }
        // 从文件中读取每帧相机的位姿。
        while(!ifPoses.eof())
        {
            double wx, wy, wz, tx, ty, tz;
            ifPoses >> wx >> wy >> wz >> tx >> ty >> tz;
            poses.push_back(mango::Pose3D(wx, wy, wz, tx, ty, tz));
        }
        ifPoses.close();
    }
    catch(std::exception e)
    {
        std::cerr << "读位姿文件异常，" << e.what() << std::endl;
        return -1;
    }

    // 每帧相机的位姿对应的每帧图像数据
    try
    {
        std::string imagesFile = dataFolder + "images.txt";
        std::ifstream ifImgs(imagesFile.c_str());
        if(!ifImgs.is_open())
        {
            std::cerr << "找不到文件：" << imagesFile << std::endl;
            return -1;
        }
        int pose_idx = 0;
        while(!ifImgs.eof())
        {
            std::string imgFile;
            ifImgs >> imgFile;
            imgFile = dataFolder + "images/" + imgFile;
            cv::Mat img = cv::imread(imgFile, cv::IMREAD_GRAYSCALE);
            run(img, poses[pose_idx++]);
        }
        ifImgs.close();
    }
    catch(std::exception e)
    {
        std::cerr << "处理图像异常，" << e.what() << std::endl;
        return -1;
    }

    return 0;
}