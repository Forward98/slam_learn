#include "harris.h"

#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../util/math_util.h"

namespace mango {

Harris::Harris() {}
Harris::~Harris() {}

void Harris::detect(const cv::Mat& src, 
                cv::Mat& resp, 
                const int aperture_size,
                const int blockSize, 
                const double k,
                Harris::ResponseType resp_type,
                cv::BorderTypes border_type)
{
    // src：输入的灰度图像。
    // aperture_size：Sobel 算子的孔径大小（奇数，通常为 3）。
    // blockSize：计算协方差矩阵时的邻域块大小（奇数）。
    // k：Harris 公式中的经验参数（通常在 [0.04, 0.06] 之间）。
    // resp_type：指定响应计算方式（Harris 公式或最小特征值）。
    // border_type：图像边界处理方式（如默认反射边界）。
    // resp：计算得到的角点响应图（归一化到 [0, 255] 的 CV_8UC1 类型）。
    cv::Size size = src.size();

    // 初始化输出响应图 resp 为与输入图像相同尺寸的零矩阵。
    resp = cv::Mat::zeros(size, resp.type());

    // 获取输入图像的尺寸。
    // 使用 OpenCV 的 Sobel 函数计算图像的横向（x 方向）和纵向（y 方向）梯度：
    // dx：x 方向梯度。
    // dy：y 方向梯度。
    // 梯度类型为 CV_32F（32 位浮点数）。
    cv::Mat dx, dy;
    // 0：x 方向的导数阶数。
    // 1：y 方向的导数阶数。
    // 孔径的作用
    // 平衡噪声敏感度和梯度精度
    // 较小的孔径（如 3×3）对噪声敏感，但能更精细地检测图像中的快速变化。
    // 较大的孔径（如 5×5、7×7）能平滑图像，减少噪声影响，但可能模糊细节。
    // 控制梯度计算的范围
    // 孔径越大，计算梯度时考虑的像素越多，适合检测较大尺度的边缘。
    // 孔径越小，计算越局部化，适合检测小尺度或精细边缘。
    cv::Sobel(src, dx, CV_32F, 1, 0, aperture_size);
    cv::Sobel(src, dy, CV_32F, 0, 1, aperture_size);

    // 创建3通道矩阵，分别存dx*dx, dx*dy, dy*dy
    cv::Mat cov(size, CV_32FC3);

    for(int i = 0; i < size.height; ++i)
    {
        float* cov_data = cov.ptr<float>(i);
        const float* dx_data = dx.ptr<float>(i);
        const float* dy_data = dy.ptr<float>(i);

        // 创建一个 3 通道的矩阵 cov，用于存储每个像素点的协方差矩阵元素：
        // 第一通道：I_x²（x 梯度的平方）。
        // 第二通道：I_xI_y（x 和 y 梯度的乘积）。
        // 第三通道：I_y²（y 梯度的平方）。
        for(int j = 0; j < size.width; ++j)
        {
            float _dx = dx_data[j];
            float _dy = dy_data[j];
            // 遍历每个像素点，填充协方差矩阵的三个通道。
            cov_data[j*3] = _dx * _dx;
            cov_data[j*3+1] = _dx * _dy;
            cov_data[j*3+2] = _dy * _dy;
        }
    }

    // 方框滤波，计算M
    // 对协方差矩阵应用方框滤波：
    // 在 blockSize x blockSize 的窗口内对每个通道进行平均。
    // 目的是在局部邻域内累加梯度信息，得到滑动窗口内的积分。
    cv::boxFilter(cov, cov, cov.depth(), cv::Size(blockSize, blockSize), cv::Point(-1, -1), false);

    // 计算响应
    cv::Mat _resp = cv::Mat::zeros(size, CV_32FC1);
    cv::Mat _resp_norm;
    // 根据响应类型计算角点响应值：
    if(resp_type == HARRIS)
    {
        calcHarris(cov, _resp, k);
    }
    else if(resp_type == MINEIGENVAL)
    {
        calcMinEigenVal(cov, _resp);
    }

    // opencv接口，测试效果是否一致
    // cv::cornerHarris(src, _resp, blockSize, aperture_size, k);

    // 归一化到[0,255]
    cv::normalize(_resp, _resp_norm, 0, 255, cv::NORM_MINMAX);
    cv::convertScaleAbs(_resp_norm, resp); // 注意这里的resp变成u8类型了
}

void Harris::getKeyPoints(const cv::Mat& resp, std::vector<cv::Point2i>& kps, const int num_kps, const int nonmaximum_sup_radius)
{
    // 这段代码是 Harris 角点检测算法中提取关键点的过程，它从角点响应图中挑选出显著的角点特征
    // 点，同时进行非极大值抑制来保证关键点的分布合理性。
    cv::Mat _resp = resp.clone();
    // 输入：
    // resp：角点响应图，每个像素的值表示该位置的角点强度。
    // nonmaximum_sup_radius：非极大值抑制的邻域半径，用于抑制局部重复关键点。
    // 输出：
    // kps：提取出的关键点列表，每个关键点是一个 cv::Point2i 类型，表示关键点在图像中的坐标。
    // 控制参数：
    // num_kps：指定最多提取的关键点数量。

    // 将输入的响应图 resp 复制到 _resp 中，避免修改原始数据。
    // 后续操作会在 _resp 上进行，以提取关键点并应用非极大值抑制。
    for(int i = 0; i < num_kps; i++)
    {
        // 循环目的：提取最多 num_kps 个关键点。
        double min_val, max_val;
        cv::Point min_idx, max_idx;
        // 寻找极值点：使用 OpenCV 的 minMaxLoc 函数找到当前响应图中的最大值及其位置 max_idx。
        cv::minMaxLoc(_resp, &min_val, &max_val, &min_idx, &max_idx);
        // 如果最大响应值小于等于 0，则停止提取关键点。这通常发生在所有显著关键点都被抑制后。
        if(max_val <= 0) break;
        // 将当前找到的最大响应点 max_idx 作为关键点添加到列表 kps 中。
        // 注意坐标顺序：OpenCV 中，max_idx.x 是列（图像的 x 轴，水平方向），max_idx.y 是行（图像的 y 轴，垂直方向）。
        kps.push_back(max_idx); // 这里是图像坐标，与行列相反
        // 邻域范围：以当前关键点为中心，nonmaximum_sup_radius 为半径的方形邻域。
        // 抑制操作：将邻域内的所有响应值置为 0，避免重复提取邻近的关键点。
        // r 和 c 分别表示行和列的偏移。
        for(int j = -nonmaximum_sup_radius; j <= nonmaximum_sup_radius; j++)
        {
            for(int k = -nonmaximum_sup_radius; k <= nonmaximum_sup_radius; k++)
            {
                int r = max_idx.y + j, c = max_idx.x + k;
                if(r >= 0 && r < _resp.rows && c >= 0 && c < _resp.cols)
                {
                    // _resp.at<uchar>(r, c) = 0：将响应值置为 0，注意 _resp 的类型是 CV_8UC1（8 位无符号整数）。
                    _resp.at<uchar>(r, c) = 0; // 注意类型uchar
                }
            }
        }
    }
}

void Harris::getKeyPoints(const cv::Mat& resp, std::vector<cv::Point2i>& kps, const double resp_threshold, const int nonmaximum_sup_radius)
{
    cv::Mat _resp = resp.clone();
    double curr_resp = .0f;

    while(1)
    {
        double min_val;
        cv::Point min_idx, max_idx;
        cv::minMaxLoc(_resp, &min_val, &curr_resp, &min_idx, &max_idx);

        if(curr_resp < resp_threshold) break;
        kps.push_back(max_idx);

        for(int j = -nonmaximum_sup_radius; j <= nonmaximum_sup_radius; j++)
        {
            for(int k = -nonmaximum_sup_radius; k <= nonmaximum_sup_radius; k++)
            {
                int r = max_idx.y + j, c = max_idx.x + k;
                if(r >= 0 && r < _resp.rows && c >= 0 && c < _resp.cols)
                {
                    _resp.at<uchar>(r, c) = 0;
                }
            }
        }
    }
}

void Harris::getDescriptors(const cv::Mat& src, const std::vector<cv::Point2i>& kps, std::vector<std::vector<uchar>>& descriptors, const int r)
{
    // 输入：
    // src：输入的灰度图像，用于提取像素值。
    // kps：关键点列表，每个关键点是一个 cv::Point2i，表示关键点在图像中的坐标。
    // r：描述符的邻域半径，描述符的大小为 (2r + 1) x (2r + 1)。
    // 输出：
    // descriptors：每个关键点对应的描述符，描述符是一个一维向量，包含关键点邻域内的像素值。
    int num_kp = (int)kps.size();
    descriptors.clear();
    descriptors.resize(num_kp);
    // 每个描述符的大小为 (2r + 1) x (2r + 1)，即邻域窗口的像素总数。例如，r = 3 时，描述符大小为 7x7=49。
    for(int i = 0; i < num_kp; i++)
    {
        descriptors[i].resize((2*r+1)*(2*r+1));
    }

    for(int i = 0; i < num_kp; i++)
    {
        int idx = 0;
        for(int j = -r; j <= r; j++)
        {
            for(int k = -r; k <= r; k++)
            {
                // 内层循环：遍历关键点周围的 (2r + 1) x (2r + 1) 邻域。
                // j 和 k 分别表示相对于关键点的行和列偏移。
                // row = kps[i].y + j 和 col = kps[i].x + k 计算当前像素的全局坐标。
                int row = kps[i].y + j, col = kps[i].x + k;
                if(row >= 0 && row < src.rows && col >= 0 && col < src.cols)
                {
                    // 将像素值存入描述符向量 descriptors[i] 中。
                    descriptors[i][idx] = src.at<uchar>(row, col);
                }
                else
                {
                    // 如果 (row, col) 在图像范围内，则提取该像素值。
                    // 如果超出图像范围，则填充为 0。
                    descriptors[i][idx] = 0;
                }

                idx++;
            }
        }    
    }
}

void Harris::match(const std::vector<std::vector<uchar>>& reference_desc, const std::vector<std::vector<uchar>>& query_desc, std::vector<int>& match_, const double lambda)
{
    // reference_desc：参考图像的特征描述子集合，每个描述子是一个向量。
    // query_desc：查询图像的特征描述子集合，每个描述子是一个向量。
    // match_：输出的匹配结果，存储每个查询点在参考图像中的匹配索引。
    // lambda：用于筛选匹配的阈值参数。

    // 获取查询图像中特征点的数量 num_kp。
    // 初始化一个向量 ssd_vec 用于存储每个查询点的最小 SSD 值。
    // 清空并调整 match_ 的大小，初始值设为 -1（表示无匹配）。
    // 初始化全局最小 SSD global_min_ssd 为一个很大的值。
    int num_kp = (int)query_desc.size();
    std::vector<double> ssd_vec(num_kp, 0);
    match_.clear();
    match_.resize(num_kp, -1);

    double global_min_ssd = std::numeric_limits<double>::max();

    // 2. 计算每个查询点的最近邻匹配
    // 遍历查询图像中的每个特征点（索引为 i）。
    // 对于每个查询点，初始化最小 SSD min_ssd 和匹配索引 match_idx。
    // 遍历参考图像中的每个特征点（索引为 j），计算查询点 i 和参考点 j 之间的 SSD。
    // 找出与查询点 i 最匹配的参考点 j（具有最小 SSD），并更新 min_ssd 和 match_idx。
    // 更新全局最小 SSD global_min_ssd。
    for(int i = 0; i < num_kp; i++)
    {
        double min_ssd = std::numeric_limits<double>::max();
        int match_idx = -1;
        for(size_t j = 0; j < reference_desc.size(); j++)
        {
            double ssd = mango::ssd<uchar>(query_desc[i], reference_desc[j]);
            if(ssd < min_ssd)
            {
                min_ssd = ssd;
                match_idx = j;

                if(min_ssd > 0 && min_ssd < global_min_ssd)
                {
                    global_min_ssd = min_ssd;
                }
            }
        }
        ssd_vec[i] = min_ssd;
        match_[i] = match_idx;
    }

    global_min_ssd *= lambda;
    // 根据阈值筛选匹配
    // 将全局最小 SSD 乘以 lambda 得到阈值。
    // 遍历所有查询点，如果某个查询点的最小 SSD 大于等于这个阈值，则将其匹配索引设为 -1（表示无匹配）。
    for(int i = 0; i < num_kp; i++)
    {
        if(ssd_vec[i] >= global_min_ssd)
        {
            match_[i] = -1;
        }
    }
}

cv::Mat Harris::plotMatchOneImage(const cv::Mat& query, const std::vector<cv::Point2i>& reference_kps, const std::vector<cv::Point2i>& query_kps, const std::vector<int>& match_)
{
    // 函数参数
    // query：查询图像。
    // reference_kps：参考图像中的关键点坐标。
    // query_kps：查询图像中的关键点坐标。
    // match_：匹配结果，表示查询图像中的每个关键点在参考图像中的匹配索引。    
    // 返回一个绘制了匹配结果的新图像 img_result。
    // 创建一个与查询图像大小相同、三通道（彩色）的空白图像，用于绘制匹配结果。
    cv::Mat img_result(query.rows, query.cols, CV_8UC3);
    if(query.channels() == 1)
    {
        std::vector<cv::Mat> channels(3, query);
        cv::merge(channels, img_result);
    }
    else
    {
        // 如果查询图像是单通道（灰度图），将其复制到三个通道以创建彩色图像。
        // 如果查询图像是彩色的，直接复制到结果图像。
        query.copyTo(img_result);
    }
    // 遍历匹配结果。
    // 如果匹配索引为 -1，跳过当前关键点。
    // 如果匹配索引有效，获取查询图像和参考图像中的对应关键点坐标。
    // 在结果图像上绘制绿色线连接匹配的关键点，并在查询图像的关键点处绘制红色圆圈。
    for(int i = 0; i < match_.size(); i++)
    {
        if(match_[i] == -1) continue;

        if(match_[i] >= 0 && match_[i] < reference_kps.size())
        {
            cv::Point2i query_kp = query_kps[i], reference_kp = reference_kps[match_[i]];
            cv::line(img_result, query_kp, reference_kp, cv::Scalar(0,255,0), 2);
            cv::circle(img_result, query_kp, 3, cv::Scalar(0,0,255), -1);
        }
    }

    return img_result;
}

cv::Mat Harris::plotMatchTwoImage(){}
    
void Harris::calcMinEigenVal(const cv::Mat& cov, cv::Mat& resp)
{
    int i, j;
    cv::Size size = cov.size();

    resp = cv::Mat::zeros(size, resp.type());

    if(cov.isContinuous() && resp.isContinuous())
    {
        size.width *= size.height;
        size.height = 1;
    }

    for(i = 0; i < size.height; ++i)
    {
        const float* cov_data = cov.ptr<float>(i);
        float* resp_data = resp.ptr<float>(i);

        for(j = 0; j < size.width; ++j)
        {
            float a = cov_data[j*3] * 0.5f;
            float b = cov_data[j*3+1];
            float c = cov_data[j*3+2] * 0.5f;
            resp_data[j] = (a + c) - std::sqrt((a - c)*(a - c) + b*b);
        }
    }
}

void Harris::calcHarris(const cv::Mat& cov, cv::Mat& resp, const double k)
{
    int i, j;
    cv::Size size = cov.size();
    
    resp = cv::Mat::zeros(size, resp.type());

    if(cov.isContinuous() && resp.isContinuous())
    {
        size.width *= size.height;
        size.height = 1;
    }

    for(i = 0; i < size.height; ++i)
    {
        const float* cov_data = cov.ptr<float>(i);
        float* resp_data = resp.ptr<float>(i);

        for(j = 0; j < size.width; ++j)
        {
            float a = cov_data[j*3];
            float b = cov_data[j*3+1];
            float c = cov_data[j*3+2];
            resp_data[j] = (float)(a*c - b*b - k*(a + c)*(a + c));
        }
    }
}
}