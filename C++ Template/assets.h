#include <vector>
#include <string>

struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    float ppoint[6]; // 标记点坐标(x,y)：左眼 & 右眼 & 鼻子
    float regreCoord[4];
};
