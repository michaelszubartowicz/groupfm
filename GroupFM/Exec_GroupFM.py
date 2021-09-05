import csv
import random
import subprocess
import numpy as np
import sys
import datetime

##############
# configuration
num_iterations = 1000
contextual_rating_effects = [1]
suffix = "1"
test_percentages = np.linspace(0.1,1,num=10,endpoint=True)
modes = ["PseudoU", "MultiU", "GC_AVG", "GC_LM", "FM_AVG", "FM_LM", "agnostic", "GC+1_AVG"]
outputname = "GroupFM"
context_hierarchy = {"time": ["morning","noon","evening"], "weather": ["warm","cold"]}
libFM_location = "..\\bin\\libFM.exe"
##############

userlist = set()
itemlist = set()
user_ratings = list()

start_time = datetime.datetime.now(datetime.timezone.utc)
def progressBar(value, endvalue, start_time, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    elapsed_time = abs(datetime.datetime.now(datetime.timezone.utc) - start_time)
    estimated_time_total = (elapsed_time / value) * endvalue
    estimated_time_remaining = estimated_time_total - elapsed_time

    sys.stdout.write("\033[K\r") # Clear to the end of line
    sys.stdout.write(f"Percent: [{arrow + spaces}] {int(round(percent * 100))}% ({value} out of {endvalue}), {elapsed_time} elapsed, {estimated_time_remaining} remaining")
    #sys.stdout.write("\033[F") # Cursor up one line
    sys.stdout.flush()

#load user ratings
with open("user_ratings.csv","r") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        userlist.add(row['user_id'])
        itemlist.add(row['restaurant_id'])
        user_ratings.append(row)
userlist = sorted(userlist)
itemlist = sorted(itemlist)

#load group ratings
grouplist = set()
group_ratings = list()
with open("group_ratings.csv","r") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        grouplist.add(row['group_id'])
        group_ratings.append(row)
grouplist = sorted(grouplist)

#load group assignments
group_assignments = dict()
with open("group_assignments.csv","r") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        group_assignments.setdefault(row['group_id'],set()).add(row['user_id'])
    
#compute all contextual situations
context_combinations = list()
def compute_context_combinations(index,current_set):
    if index < len(context_hierarchy.keys()):
        ctx_var = list(context_hierarchy.keys())[index]
        for ctx_val in context_hierarchy[ctx_var]:
            new_set = list()
            new_set.append(ctx_var + "_" + ctx_val)
            new_set.extend(current_set)
            compute_context_combinations(index+1,new_set)
    else:
        context_combinations.append(current_set)
            
compute_context_combinations(0,[])   

def run(contextual_rating_effect,mode,test_percentage,iteration,suffix):
    aggregated_lines_count = list()

    #setting indices for items and context features
    context_indices = dict()
    item_feature_count_offset = len(userlist)
    if mode == "PseudoU":
        item_feature_count_offset += len(grouplist)
    elif mode in ["GC_AVG","GC_LM"]:
        item_feature_count_offset += len(userlist)
    context_feature_count_offset = item_feature_count_offset + len(itemlist)
    for ctx_var in context_hierarchy.keys():
        for ctx_val in context_hierarchy[ctx_var]:
            context_indices[ctx_var + "_" + ctx_val] = context_feature_count_offset
            context_feature_count_offset += 1

    # contextualize ratings and generate output in libFM format
    def transformlines(inputlines):
        outputlines = list()
        for row in inputlines:
            if row["rating"] == '':
                continue
            baserating = int(row["rating"])
            feature_values = dict()

            if "group_id" in row:
                feature_values[row["group_id"]] = 1
            else:
                user = row["user_id"]
                feature_values[userlist.index(user)] = 1

            item = row["restaurant_id"]
            feature_values[itemlist.index(item) + item_feature_count_offset] = 1

            for context_set in context_combinations:
                rating = baserating
                current_ctx_indices = list()
                context_is_captured = True
                for context in context_set:
                    current_ctx_indices.append(context_indices[context])
                    if row[context] == '-':
                        rating -= contextual_rating_effect
                    elif row[context] == '+':
                        rating += contextual_rating_effect
                    elif row[context] != '0':
                        context_is_captured = False
                        break
                rating = min(max(rating,1),5)
                
                if context_is_captured:
                    string_components = list()
                    string_components.append(str(rating))
                    for k,v in feature_values.items():
                        string_components.append(str(k)+":"+str(v))
                    for ctx_index in current_ctx_indices:
                        string_components.append(str(ctx_index)+":1")
                    outputlines.append(" ".join(string_components) + "\n")
        return outputlines

    def handlegroups(inputlines,isTestData):
        outputlines = []
        for row in inputlines:
            all_feature_values = list()
            
            #determine the group feature which will be replaced
            parts = row.split(" ")
            part_index = -1
            group = ""
            for part in parts:
                if part.startswith("G"):
                    group = part.split(":")[0]
                    part_index = parts.index(part)

            if mode == "PseudoU":
                feature_values = dict()
                feature_values[grouplist.index(group) + len(userlist)] = 1
                all_feature_values.append(feature_values)
            elif mode == "agnostic":
                feature_values = dict()
                all_feature_values.append(feature_values)
            elif mode == "MultiU":
                group_members = group_assignments[group]
                feature_values = dict()
                for member in sorted(group_members):
                    feature_values[userlist.index(member)] = 1/len(group_members)
                all_feature_values.append(feature_values)
            elif mode in ["GC_AVG","GC_LM","FM_AVG","FM_LM","GC+1_AVG"]:
                group_members = group_assignments[group]
                if isTestData:
                    aggregated_lines_count.append(len(group_members))
                elif mode.startswith("FM"):
                    #baseline model is built on single user data only
                    continue
                for member in sorted(group_members):
                    feature_values = dict()
                    feature_values[userlist.index(member)] = 1
                    if mode.startswith("GC"):
                        others = group_members.difference([member])
                        for other in sorted(others):
                            if mode == "GC+1_AVG":
                                feature_values[userlist.index(other) + len(userlist)] = 1/(len(others)+1)
                            else:
                                feature_values[userlist.index(other) + len(userlist)] = 1/len(others)
                    all_feature_values.append(feature_values)
            else:
                continue

            #override group component of current row
            for feature_values in all_feature_values:
                new_components = list()
                for k,v in feature_values.items():
                    new_components.append(str(k)+":"+str(v))
                parts[part_index] = " ".join(new_components)
                outputlines.append(" ".join(parts))

        return outputlines

    userdata_formatted = transformlines(user_ratings)
    groupdata_processed = transformlines(group_ratings)

    #create testratings
    num_test_rows = 250
    num_erased_rows = round(float(test_percentage)*len(groupdata_processed)) - num_test_rows
    testdata_raw = random.sample(groupdata_processed,num_test_rows)
    for row in testdata_raw:
        groupdata_processed.remove(row)
    erased_rows = random.sample(groupdata_processed,num_erased_rows)
    for row in erased_rows:
        groupdata_processed.remove(row)

    groupdata_formatted = handlegroups(groupdata_processed,False)
    testdata = handlegroups(testdata_raw,True)
    traindata = userdata_formatted + groupdata_formatted

    #write files so libFM can read them
    with open(f"./run/train_{outputname}_{suffix}.libfm","w", newline='') as outfile:
        outfile.writelines(traindata)

    with open(f"./run/test_{outputname}_{suffix}.libfm","w", newline='') as outfile:
        outfile.writelines(testdata)

    #run libFM
    subprocess.run([libFM_location,"-method","als","-train",f"./run/train_{outputname}_{suffix}.libfm","-test",f"./run/test_{outputname}_{suffix}.libfm","-out",f"./run/output_{outputname}_{suffix}.txt","-task","r","-seed",str(iteration), "-verbosity", "0"], stdout=subprocess.DEVNULL)

    #calculate RMSE
    predictions = list()
    targets = list()
    with open(f"./run/output_{outputname}_{suffix}.txt","r") as infile:
        for row in infile.readlines():
            predictions.append(float(row))

    #aggregate predictions if necessary
    if mode in ["GC_AVG","GC_LM","FM_AVG","FM_LM","GC+1_AVG"]:
        aggregated_predictions = list()
        prediction_index = 0
        for current_count in aggregated_lines_count:
            current_predictions = predictions[prediction_index:prediction_index+current_count]
            if mode.endswith("AVG"):
                aggregated_predictions.append(float(np.mean(current_predictions)))
            elif mode.endswith("LM"):
                aggregated_predictions.append(min(current_predictions))
            prediction_index+=current_count
        predictions = aggregated_predictions

    for row in testdata_raw:
        targets.append(float(row.split(' ')[0]))

    errors = np.subtract(predictions,targets)
    rmse = np.sqrt((errors ** 2).mean())

    return rmse

total_runs = num_iterations*len(contextual_rating_effects)*len(test_percentages)*len(modes)
current_run = 0
output_dicts = list()
for contextual_rating_effect in contextual_rating_effects:
    for test_percentage in test_percentages:
        currentdict = dict()
        currentdict['contextual_rating_effect'] = str(contextual_rating_effect)
        currentdict['test_percentage'] = str(test_percentage)
        for mode in modes:
            rmse = 0
            for i in range(num_iterations):
                random.seed(i)
                rmse += run(contextual_rating_effect,mode,test_percentage,i,suffix)
                current_run += 1
                progressBar(current_run,total_runs,start_time)
            rmse /= num_iterations
            currentdict[mode] = str(rmse)
        output_dicts.append(currentdict)
print('\n')

with open(f'results_{outputname}_{suffix}.csv', 'w', newline='') as csvfile:
    fieldnames = ['contextual_rating_effect','test_percentage',"PseudoU", "MultiU", "GC_AVG", "GC_LM", "GC+1_AVG", "agnostic", "FM_AVG", "FM_LM"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()
    writer.writerows(output_dicts)
