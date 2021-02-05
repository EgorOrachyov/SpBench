import datetime
import csv


def parse_datetime(datetime_str):
    return datetime.datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S.%f")


def load_time_points(path):
    experiments_list = []
    with open(path, "r") as file:
        lines = file.readlines()
        count = len(lines)
        for i in range(0, count, 3):
            names = lines[i].replace("\n", "").split(' ')
            experiment = names[0] + " " + names[1]
            begin = parse_datetime(lines[i + 1].replace("\n", ""))
            end = parse_datetime(lines[i + 2].replace("\n", ""))
            experiments_list.append((experiment, begin, end))

    return experiments_list


def load_mem_profile(path):
    table_list = []
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        header = next(reader, None)
        # print("Import table with header: ", header)
        for row in reader:
            table_list.append((parse_datetime(row[0]), row[1], int(row[3].replace("MiB", ""))))

    return table_list


def find_nearest(v, d, left):
    size = len(d)
    l = 0
    r = size - 2

    while l <= r:
        m = int((l + r) / 2)
        if d[m][0] <= v <= d[m + 1][0]:
            return m if left else m + 1
        elif d[m + 1][0] <= v:
            l = m + 1
        else:
            r = m - 1

    return None


def extract_sections(experiments_list, table_list):
    sections_list = []
    for e in experiments_list:
        begin = find_nearest(e[1], table_list, False)
        end = find_nearest(e[2], table_list, True)
        sections_list.append((begin, end))

    return sections_list


def extract_stats_per_section(sections_list, table_list):
    stats_list = []
    for section in sections_list:
        i, j = section[0], section[1]
        min_mem = 1e10
        max_mem = 0
        for c in range(i, j):
            mem = table_list[c][2]
            min_mem = min(min_mem, mem)
            max_mem = max(max_mem, mem)
        stats_list.append((min_mem, max_mem))

    return stats_list


def output_stats_to_file(path, experiments_list, stats_list):
    with open(path, "w") as file:
        n = len(experiments_list)
        assert n == len(stats_list)

        for i in range(n):
            min_mem = stats_list[i][0]
            max_mem = stats_list[i][1]
            file.write(f'[{i}] {experiments_list[i][0]}\n')
            file.write(f'min= {min_mem} MiB max= {max_mem} MiB used= {max_mem - min_mem} MiB\n\n')


mem_profileF_fle = "Profiling.csv"
mem_profile_time_file = "Profiling-Time.txt"
mem_profile_stats_file = "Profiling-Stats.txt"

experiments = load_time_points(mem_profile_time_file)
table = load_mem_profile(mem_profileF_fle)
sections = extract_sections(experiments, table)
stats = extract_stats_per_section(sections, table)
output_stats_to_file(mem_profile_stats_file, experiments, stats)

# print(experiments)
# print(table)
# print([(i, sections[i]) for i in range(len(sections))])
# print([(i, table[sections[i][0]], table[sections[i][1]]) for i in range(len(sections))])
# print(stats)