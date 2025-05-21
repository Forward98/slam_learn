#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    // 指定图片路径
    string path = "E:\\VScode_project\\slam_learn\\src\\img_0067.png";
    cv::Mat img = imread(path);  // 读取图像

    // 显示图像
    imshow("Display Image", img);
    cout << "Image size: " << img.size() << endl;  // 输出图像大小
    // imshow("Display Image", test);

    // 等待按键按下，关闭窗口
    waitKey(0);

    return 0;
}
