#include <stdio.h>
#include <stdlib.h>

#define MAX_LEN 128

typedef struct {
    int id;
    char name[MAX_LEN];
} User;

/* 计算两个整数的和 */
int add(int a, int b) {
    return a + b;
}

static int clamp(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

char *make_name(const char *prefix, int id) {
    char *buf = malloc(MAX_LEN);
    if (!buf) return NULL;
    snprintf(buf, MAX_LEN, "%s-%d", prefix, id);
    return buf;
}

void print_user(const User *u) {
    if (!u) return;
    printf("id=%d name=%s\n", u->id, u->name);
}

int main(void) {
    User u = {1, "alice"};
    print_user(&u);
    printf("%d\n", add(1, 2));
    return 0;
}
