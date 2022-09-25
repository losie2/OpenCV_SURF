//
//  1-2.c
//  Practice
//
//  Created by 권오준 on 2022/09/25.
//

#include <stdio.h>

int main(void) {
    int property;

    printf("매달 저축 금액을 입력하시오: ");
    scanf_s("%d", &property);

    printf("30년 후의 재산 = %d원\n", property * 12 * 30);

    return 0;
}
