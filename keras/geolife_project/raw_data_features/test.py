def main():
    my_list = [[(5, 10, 'a'), (11, 20, 'b'), (23, 50, 'c'), (51, 60, 'd')], [(5, 10, 'e'), (11, 20, 'f'), (23, 50, 'g'), (51, 60, 'h')]]
    trajs = [list(range(0, 71)), list(range(0, 71))]

    for u in range(len(trajs)):
        i = 0
        j = 0
        while (i < len(trajs[u])) and (j < len(my_list[u])):
            
            left = my_list[u][j][0]
            right = my_list[u][j][1]
            label = my_list[u][j][2]

            num = trajs[u][i]

            if (left <= num <= right):
                print(num, label)
                i += 1
            elif (num < left):
                print(num, 'NaN')
                i += 1
            else:
                j += 1

if __name__ == '__main__':
    main()