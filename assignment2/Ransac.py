# 2. 【Reading + Pseudo Code】
#       We haven't told RANSAC algorithm this week. So please try to do the reading.
#       And now, we can describe it here:
#       We have 2 sets of points, say, Points A and Points B. We use A.1 to denote the first point in A,
#       B.2 the 2nd point in B and so forth. Ideally, A.1 is corresponding to B.1, ... A.m corresponding
#       B.m. However, it's obvious that the matching cannot be so perfect and the matching in our real
#       world is like:
#       A.1-B.13, A.2-B.24, A.3-x (has no matching), x-B.5, A.4-B.24(This is a wrong matching) ...
#       The target of RANSAC is to find out the true matching within this messy.
#
#       Algorithm for this procedure can be described like this:
#       1. Choose 4 pair of points randomly in our matching points. Those four called "inlier" (中文： 内点) while
#          others "outlier" (中文： 外点)
#       2. Get the homography of the inliers
#       3. Use this computed homography to test all the other outliers. And separated them by using a threshold
#          into two parts:
#          a. new inliers which is satisfied our computed homography
#          b. new outliers which is not satisfied by our computed homography.
#       4. Get our all inliers (new inliers + old inliers) and goto step 2
#       5. As long as there's no changes or we have already repeated step 2-4 k, a number actually can be computed,
#          times, we jump out of the recursion. The final homography matrix will be the one that we want.
#
#       [WARNING!!! RANSAC is a general method. Here we add our matching background to that.]
#
#       Your task: please complete pseudo code (it would be great if you hand in real code!) of this procedure.
#
#       Python:
#       def ransacMatching(A, B):
#           A & B: List of List
#
# //      C++:
# //      vector<vector<float>> ransacMatching(vector<vector<float>> A, vector<vector<float>> B) {
# //      }
#
#       Follow up 1. For step 3. How to do the "test“? Please clarify this in your code/pseudo code
#       Follow up 2. How do decide the "k" mentioned in step 5. Think about it mathematically!

"""
资料：https://www.cnblogs.com/weizc/p/5257496.html <br>

在图像拼接中用于估计单应性矩阵：
1. 随机从数据集中抽出4个样本数据 (此4个样本之间不能共线)，则这4对点叫做内点（inlier),其它的点叫外点(outlier);
2. 计算出单应性矩阵H，记为模型M；
3. 计算所有outliers与模型M的投影误差，若误差小于阈值，加入内点集 I ；
4. 如果当前内点集 I 元素个数大于最优内点集 I_best , 则更新 I_best = I，同时更新迭代次数k ;
5. 如果迭代次数大于k,则退出 ; 否则迭代次数加1，并重复上述步骤；

伪代码：
input:
pass
output:
pass

iter = 0
max_iter = 2000
best_homography = None
best_points_set = None
best_error = 无穷大

while(iter < max_iter):
    maybe_inliers = 从点集中随机选取四对点
    maybe_homography = 从maybe_inliers估计出的homography
    inliers = maybe_inliers

    for(outliers中的点)：
        if(点适用于maybe_homography and 错误小于threshold):
            将点添加到inliers
    if(inliers中的点的数目大于d)：
        best_homography = 适合于inliers中的点的homograpyh
        error = 误差度量
        if(error < best_error):
            best_homography = mayby_homography
            best_points_set = inliers
            best_error = error
    iter += 1
返回 best_model, best_points_set, best_error

"""