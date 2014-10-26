load('GData');

points1 = f1(1:2,idx(1,:))';
points2 = f2(1:2,idx(2,:))';

H_mle = homography_estimator(points1, points2, 'mlesac');
H_pro = homography_estimator(points1, points2, 'prosac');
H_ran = homography_estimator(points1, points2, 'ransac');

disp('Homography estimated using MLESAC:'); H_mle

disp('Homography estimated using RANSAC:'); H_ran

disp('Homography estimated using PROSAC:'); H_pro