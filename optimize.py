#PYTHON 
def first_fit_decreasing(n, k, items, trucks):
    # Sort items by area in decreasing order
    items.sort(key=lambda x: x[0]*x[1], reverse=True)

    # Initialize trucks
    truck_areas = [(i+1, w*l, w, l, c, []) for i, (w, l, c) in enumerate(trucks)]
    truck_areas.sort(key=lambda x: x[1], reverse=True)

    # Initialize solution
    solution = [None] * n

    for item in items:
        w, l = item[:2]
        rotated = False
        for truck in truck_areas:
            t, area, tw, tl, c, items_in_truck = truck
            if (w <= tw and l <= tl) or (w <= tl and l <= tw):
                if w <= tw and l <= tl:
                    for x in range(tw - w + 1):
                        for y in range(tl - l + 1):
                            if all(x >= x_in_truck + w_in_truck or x + w <= x_in_truck or y >= y_in_truck + l_in_truck or y + l <= y_in_truck for x_in_truck, y_in_truck, w_in_truck, l_in_truck, o_in_truck in items_in_truck):
                                items_in_truck.append((x, y, w, l, 0))
                                solution[item[2]-1] = (item[2], t, x, y, 0)
                                break
                        else:
                            continue
                        break
                else:
                    for x in range(tw - l + 1):
                        for y in range(tl - w + 1):
                            if all(x >= x_in_truck + w_in_truck or x + l <= x_in_truck or y >= y_in_truck + l_in_truck or y + w <= y_in_truck for x_in_truck, y_in_truck, w_in_truck, l_in_truck, o_in_truck in items_in_truck):
                                items_in_truck.append((x, y, l, w, 1))
                                solution[item[2]-1] = (item[2], t, x, y, 1)
                                break
                        else:
                            continue
                        break

    return solution

n, k = map(int, input().split())
items = []
for i in range(n):
    w, l = map(int, input().split())
    items.append((w, l, i+1))

trucks = []
for i in range(k):
    w, l, c = map(int, input().split())
    trucks.append((w, l, c))

solution = first_fit_decreasing(n, k, items, trucks)

for item in solution:
    if item is not None:
        print(' '.join(map(str, item)))
    else:
        print("0 0 0 0 0")
