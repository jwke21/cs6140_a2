"""
CS6140 Project 2
Jake Van Meter
Yihan Xu
"""
import sys
from nn import first_task
from cluster import second_task, third_task
from knn import fourth_task


def main():
    argv = sys.argv
    # If desired task is not specified, run all tasks
    if len(argv) == 1:
        argv.append("all")
    #################### PART 1 ####################
    if "all" in argv or "1" in argv:
        first_task()
    #################### PART 2 ####################
    if "all" in argv or "2" in argv:
        second_task()
    #################### PART 3 ####################
    if "all" in argv or "3" in argv:
        third_task()
    #################### PART 4 ####################
    if "all" in argv or "4" in argv:
        fourth_task()

if __name__ == "__main__":
    main()
