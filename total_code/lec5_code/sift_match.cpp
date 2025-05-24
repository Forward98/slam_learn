#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
// #include <opencv2/features2d.hpp>

#include "feature/sift.h"
#include "util/string_util.h"
#include "util/image_util.h"

namespace mango {

class SiftMatch
{
public:
    SiftMatch(const std::string& data_folder)
        : data_folder_(data_folder)
    {}
    ~SiftMatch() {}

    void run();

private:
    std::string data_folder_;
};

void SiftMatch::run()
{
    try
    {
        mango::Sift sift(200, 5, 3, 0.04, 10, 1.6);
        cv::Mat image1 = cv::imread(data_folder_ + "img1.png", cv::IMREAD_GRAYSCALE);
        cv::Mat image2 = cv::imread(data_folder_ + "img2.png", cv::IMREAD_GRAYSCALE);

        std::vector<cv::KeyPoint> kp_ref;
        std::vector<std::vector<float>> desc_ref;
        sift.detect(image1, kp_ref, desc_ref);
        sift.plotGaussianPyramid("E:/VScode_project/slam_learn/output/gaussian_pyr_ref.png", true);
        sift.plotDogPyramid("E:/VScode_project/slam_learn/output/dog_pyr_ref.png", true);
        sift.plotKeypoints(image1, kp_ref, "E:/VScode_project/slam_learn/output/kp_ref.png", true);

        std::vector<cv::KeyPoint> kp_query;
        std::vector<std::vector<float>> desc_query;
        sift.detect(image2, kp_query, desc_query);
        sift.plotGaussianPyramid("E:/VScode_project/slam_learn/output/gaussian_pyr_query.png", true);
        sift.plotDogPyramid("E:/VScode_project/slam_learn/output/dog_pyr_query.png", true);
        sift.plotKeypoints(image2, kp_query, "E:/VScode_project/slam_learn/output/kp_query.png", true);

        std::vector<int> match_idx;
        sift.match(desc_ref, desc_query, match_idx);
        sift.plotMatchTwoImage(image1, image2, kp_ref, kp_query, match_idx, "E:/VScode_project/slam_learn/output/match.png", true);
    }
    catch(std::exception e)
    {
        std::cerr << "image process exception: " << e.what() << std::endl;
    }
}

}

int main(int argc, char** argv)
{
    // if(argc != 2)
    // {
    //     std::cout << "Usage: sift_match ../data" << std::endl;
    //     return 0;
    // }

    std::string data_folder = "E:/VScode_project/slam_learn/data/"; // 绝对路径

    mango::SiftMatch sift_match(data_folder);
    sift_match.run();

    return 0;
}