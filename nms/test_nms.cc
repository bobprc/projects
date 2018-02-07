#include <iostream>
#include <vector>
#include <random>

using namespace std;

vector<int> nms(const vector<vector<float>>&,
                const vector<float>&,
                const float);

int main()
{
  vector<vector<float>> boxes;
  vector<float> scores;
  default_random_engine e1(0);
  normal_distribution<float> jitter(0, 10);
  for(int i=0; i<10; ++i)
  {
    vector<float> box{150 + jitter(e1),
                      200 + jitter(e1),
                      30 + jitter(e1),
                      100 + jitter(e1)};
    boxes.push_back(box);
    scores.push_back(jitter(e1));
  }
  for(int i=0; i<10; ++i)
  {
    vector<float> box{1000 + jitter(e1),
                      560 + jitter(e1),
                      300 + jitter(e1),
                      100 + jitter(e1)};
    boxes.push_back(box);
    scores.push_back(jitter(e1));
  }
  for(int i=0; i<10; ++i)
  {
    vector<float> box{100 + jitter(e1),
                      2000 + jitter(e1),
                      300 + jitter(e1),
                      200 + jitter(e1)};
    boxes.push_back(box);
    scores.push_back(jitter(e1));
  }

  vector<int> mask = nms(boxes, scores, 0.1);

  for(int m: mask)
    cout << m << "\n";
}
