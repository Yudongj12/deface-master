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
    float ppoint[6]; // ��ǵ�����(x,y)������ & ���� & ����
    float regreCoord[4];
};
