#include "file_util.h"

#include <iostream>

namespace mango {

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> load_matrix(const std::string& file, int row, int col)
{

    // 参数 120 和 24 表示文件中数据的维度，即矩阵的行数和列数。
    // 这表明 detected_corners.txt 文件中存储的数据是一个 120 行 24 列的矩阵，
    // 每一行数据代表12个3D坐标点在二维图像中的投影点的x,y点, 这些数据代表了角点在图像中的投影像素坐标。
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;
    mat.resize(row, col);

    std::ifstream fs(file.c_str());
    if(!fs.is_open())
    {
        std::cerr << "failed to load file: " << file << std::endl;
        return mat;
    }
    for(int r = 0; r < row; r++)
    {
        for(int c = 0; c < col; c++)
        {
            fs >> mat(r, c);
        }
    }

    fs.close();
    return mat;
}

}