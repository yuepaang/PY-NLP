#include <unistd.h>
#include <stdio.h>

int main(int argc, char const *argv[])
{
    int pid;
    pid = fork();    // 使用 fork 函数

    if (pid < 0) {
        printf("Fail to create process\n");
    }
    else if (pid == 0) {
        printf("I am child process (%d) and my parent is (%d)\n", getpid(), getppid());
    }
    else {
        printf("I (%d) just created a child process (%d)\n", getpid(), pid);
    }
    return 0;
}
