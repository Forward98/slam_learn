#include <iostream>
#include "struct_.h"
using namespace std;

void struct_(){
    struct student
    {
        /* data */
        int num;
        char name[20];
        char gender;
    };
    //结构体内存对齐
    student s = {10, "asd", 'M'};
    cout << s.num << endl;
    cout << sizeof(s.num) << endl;
    cout << sizeof(s.name) << endl;
    cout << sizeof(s.gender) << endl;
    cout << sizeof(s) << endl;
    /*
    输出结果为：
    10
    4
    20
    1
    28

    结构体内存对齐需要满足的基本原则：
    1) 结构体变量的首地址能够被其最宽基本类型成员的大小所整除；
    2) 结构体每个成员相对结构体首地址的偏移量(offset)都是成员大小的整数倍，如有需要编译器
    会在成员之间加上填充字节;
    3) 结构体的总大小为结构体最宽基本类型成员大小的整数倍，如有需要编译器会在最末一个成员之后加上填充字节
    */
}
