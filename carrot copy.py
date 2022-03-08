import math
for i in range(3):
    w, h, r = input().split()
    w = int(w)
    h = int(h)
    r = int(r)
    point = 0
    incircle = 0
    incircle_zero = 0
    incircle_one = 0
    incircle_two = 0
    incircle_thr = 0
    # 직교좌표계 기준 x, y != 0인 사분원 내 점 개수
    for y in range(1, r): # y == 1 부터 y == 까지
        cosseta = math.sqrt((r*r) - (y*y))/r
        incircle += math.floor((r * cosseta))
    
    # 원 안에 총 점 개수
    incircle = incircle*4
    x0y0 = (r*4+1)
    #  마굿간 범위 제외 점 개수 새기 (Case 분류)
    if (r < w) and (r < h): # r == 3인 케이스 (첫 입력값)
        point = ((incircle/4)*3) + (x0y0 - (r*2 +1))
        point = int(point)

    elif (r<w) and (r>h): # r == 9인 케이스 (예제 확인용)
        for y in range(1, r-h):
            cosseta_zero = math.sqrt(((r-h)*(r-h)) - (y*y))/(r-h)
            incircle_zero += math.floor((r * cosseta_zero))
            point = ((incircle/4)*3) + incircle_zero + x0y0 - (1+r+(r-h))

    elif w+h > r:# r == 15인케이스 (두번째 입력값)
        for y in range(1, r-h):
            cosseta_one = math.sqrt(((r-h)*(r-h)) - (y*y))/(r-h)
            incircle_one += math.floor(((r-h) * cosseta_one))
        for y in range(1, r-w): 
            cosseta_two = math.sqrt(((r-w)*(r-w)) - (y*y))/(r-w)
            incircle_two += math.floor(((r-w) * cosseta_two))
        point = ((incircle/4)*3) + incircle_one + incircle_two + (x0y0 - (h+w+1))
        point = int(point)

    elif w+h < r: # r == 20인 케이스(세번째 입력값)
        for y in range(1, r-(h+w)):
            cosseta_thr = math.sqrt(((r-(h+w))*(r-(h+w))) - (y*y))/(r-(h+w))
            incircle_thr += math.floor(((r-(h+w)) * cosseta_thr))
        point = (incircle + (x0y0 - ((w*h))) - (incircle_thr*2))
        point = int(point)
    print(point)

