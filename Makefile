#### CCS 2017 EXPERIMENTATION AND BENCHMARKS

#### Additional arXiv version experiments and benchmarks are included
#### in the second half of this Makefile. The extended version is
#### available at https://arxiv.org/abs/1705.07807 .

package:
	cd ..; tar -cvf artifact.tar artifact; gzip artifact.tar

clean:
	rm -Rf *.pyc *.tsv *.csv *.pdf

### Section 6.1 (Example Workflow)

## Figure 3

WORKFLOW1_ARGS := data/adult/adult_cleaner.csv --sub_sample 1000 --no_normalize \
	-s marital_status -c earning --remove_sensitive \
	-m decision-tree --nominal_encoding single --max_depth 5 --criterion gini

workflow1.base.tsv workflow1.arrows.tsv workflow1.anchors.tsv: *.py 
	python -u gen_interp.py $(WORKFLOW1_ARGS) \
		--save_output1 workflow1.base.tsv \
		--save_output2 workflow1.arrows.tsv \
		--save_output3 workflow1.anchors.tsv \
		--color blue --label "exps."

workflow1: *.py \
		workflow1.base.tsv \
		workflow1.arrows.tsv \
		workflow1.anchors.tsv \
		annots/workflow1.add_labels.tsv
	python -u plot_interp.py \
		--input1 workflow1.base.tsv    \
		--input2 workflow1.arrows.tsv  \
		--input3 workflow1.anchors.tsv annots/workflow1.add_labels.tsv \
		--show --output workflow1.pdf --bw

## Figure 4

WORKFLOW2_ARGS:= $(WORKFLOW1_ARGS) --remove relationship

workflow2.base.tsv workflow2.arrows.tsv workflow2.anchors.tsv: *.py 
	python -u gen_interp.py $(WORKFLOW2_ARGS) \
		--save_output1 workflow2.base.tsv \
		--save_output2 workflow2.arrows.tsv \
		--save_output3 workflow2.anchors.tsv \
		--color blue --label "exps."

workflow2.repaired.base.tsv workflow2.repaired.arrows.tsv workflow2.repaired.anchors.tsv: *.py
	python -u gen_repair_and_interp.py \
		$(WORKFLOW2_ARGS) \
		--epsilon 0.06 --delta 0.08 --subrepair \
		--save_output1 workflow2.repaired.base.tsv \
		--save_output2 workflow2.repaired.arrows.tsv \
		--save_output3 workflow2.repaired.anchors.tsv \
		--color green --label "exps. (repaired)"

workflow2: *.py \
		workflow2.base.tsv \
		workflow2.arrows.tsv \
		workflow2.anchors.tsv \
		workflow2.repaired.base.tsv \
		workflow2.repaired.arrows.tsv \
		workflow2.repaired.anchors.tsv \
		annots/workflow2.add_labels.tsv
	python -u plot_interp.py \
		--input1 workflow2.base.tsv    workflow2.repaired.base.tsv \
		--input2 workflow2.arrows.tsv  workflow2.repaired.arrows.tsv \
		--input3 workflow2.anchors.tsv workflow2.repaired.anchors.tsv annots/workflow2.add_labels.tsv \
		--show --output workflow2.pdf --bw

# Proxy:
#ite(sex ≤ 0.500000,
#  ite(capital_loss ≤ 1822.500000,
#    0,
#    ite(age ≤ 31.500000,
#      0,
#      1)
#  )
#,
#  ite(age ≤ 29.500000,
#    ite(hours_per_week ≤ 55.000000,
#      0,
#      1)
#  ,
#    1)
#)

violations_adult: *.py
	python subexp_stats.py $(WORKFLOW2_ARGS) --epsilon 0.08 --delta 0.08 --verbose

### Section 6.2 (Other Case Studies)

CASES_ARGS := python -u subexp_stats.py --remove_sensitive --show_figure --no_normalize --verbose

## Targeted contraception advertising (details in Appendix D.1)

# Proxy 1: education ≤ 3.50
# Proxy 2:
#   ite(education ≤ 3.50,
#     ite(children ≤ 2.50,
#       ite(age ≤ 30.50,
#         1,
#         1)
#      ...
# Other proxies: Variations using the same features as above.

violations_contra: *.py
	$(CASES_ARGS) data/nics/cmc.csv \
		-s religion -c contra \
		-m decision-tree --max_depth 5 \
		--epsilon 0.005 --delta 0.05

## Student assistance (details in Appendix D.1)

# Note that studytime attribute has a non-obvious encoding. See
# data/sac/student.names for details.

# Proxy 1
#   studytime ≤ 1.50
# Proxy 2
#   ite(studytime ≤ 1.50,
#     ite(Fedu ≤ 3.50,
#       0.00,
#       1.00)
#   ,
#     ite(absences ≤ 8.50,
#       1.00,
#       0.00)
#   )
# Some other proxies.

violations_student: *.py
	$(CASES_ARGS) data/sac/student-processed.csv \
		-s Walc -c Grade \
		-m decision-tree --max_depth 5 \
		--epsilon 0.01 --delta 0.05

## Credit advertisements (details in Appendix D.1)

# Model 1 (target based on student loan)
# Note, output=5 has student loan, and output=1 does not.
# Proxy
#   ite(children ≤ 1.50,
#     ite(work_hours ≤ 37.50,
#       1,
#       5)
#   ,
#     ite(auto_insurance ≤ 105.50,
#       5,
#       5)
#   )
# Some other proxies.

violations_credit_student_loans: *.py
	$(CASES_ARGS) data/psid/fam_credit.csv \
		-s health_status -c stud_loan \
		-m decision-tree --max_depth 7 \
		--epsilon 0.015 --delta 0.015

# Model 2 (target based on credit card)

# Proxy
#    income ≤ 33315.00

violations_credit_existing_credit: *.py
	$(CASES_ARGS) data/psid/fam_credit.csv \
		-s health_status -c credit_card \
		-m decision-tree --max_depth 5 \
		--epsilon 0.01 --delta 0.05


### Section 6.3 (Detection and Repair)

# Figure 5 (runtime vs. dataset size)

BENCH_PARAMS := python -u bench_detect.py \
	data/adult/adult_cleaner.csv -s marital_status -c earning \
	--seed 0 --nominal_encoding single --remove_sensitive \
	--show_figure --association nmi --epsilon 0.0 --delta 0.0

BD11df := data_bench_detect_adult_vs_dataset_tree.tsv
$(BD11df): *.py
	$(BENCH_PARAMS) -m decision-tree --max_depth 5 > $(BD11df)

BD12df := data_bench_detect_adult_vs_dataset_forest.tsv
$(BD12df): *.py
	$(BENCH_PARAMS)	-m random-forest --forest_trees 3 --max_depth 5	> $(BD12df)

BD13df := data_bench_detect_adult_vs_dataset_logistic.tsv
$(BD13df): *.py
	$(BENCH_PARAMS)	-m logistic --reg_param 0.000385 > $(BD13df)

plot_bench_dataset: *.py $(BD11df) $(BD12df) $(BD13df)
	python -u plot_bench_detect.py --input1 $(BD11df) $(BD12df) $(BD13df)\
		--show --output plot_bench_detect_vs_dataset_all.pdf

## Figure 6 (repair accuracy vs. influence)

exp_repair_random_sac: *.py
	python -u exp_repair_random.py data/sac/student-processed.csv -s Walc -c Grade \
		-m decision-tree --max_depth 7 --train_sensitive --sensitive_max_depth 8 --no_normalize \
		--epsilon 0.01:0.25:0.01 --delta 0.01:0.25:0.01 > data_exp_repair_random_5_8_sac.tsv
plot_repair_random_sac: *.py
	python -u plot_repair_random.py --input1 data_exp_repair_random_5_8_sac.tsv \
		--show --output plot_repair_random_sac.pdf

#### arXiv Version
# The following experiments are not present in the CCS submission due
# to space reasons. The Figure and Section numbers below refer to the
# extended version.

#### Appendix D (Other Experiments)

## Figure 8

OMNIBUS_ARGS := python -u gen_interp.py \
	data/adult/adult_cleaner.csv --sub_sample 1000 \
	-s marital_status -c earning --remove_sensitive --nominal_encoding single

omnibus.tree.base.tsv omnibus.tree.arrows.tsv omnibus.tree.anchors.tsv:
	$(OMNIBUS_ARGS) -m decision-tree --max_depth 5 --criterion gini \
		--save_output1 omnibus.tree.base.tsv \
		--save_output2 omnibus.tree.arrows.tsv \
		--save_output3 omnibus.tree.anchors.tsv \
		--color blue --label "decision tree"

omnibus.logistic.base.tsv omnibus.logistic.arrows.tsv omnibus.logistic.anchors.tsv:
	$(OMNIBUS_ARGS) -m lasso --reg_param 0.08 \
		--save_output1 omnibus.logistic.base.tsv \
		--save_output2 omnibus.logistic.arrows.tsv \
		--save_output3 omnibus.logistic.anchors.tsv \
		--color red --label "logistic"

omnibus.forest.base.tsv omnibus.forest.arrows.tsv omnibus.forest.anchors.tsv:
	$(OMNIBUS_ARGS) -m random-forest --forest_trees 3 --max_depth 5 --criterion gini \
		--save_output1 omnibus.forest.base.tsv \
		--save_output2 omnibus.forest.arrows.tsv \
		--save_output3 omnibus.forest.anchors.tsv \
		--color green --label "random forest"

omnibus: *.py \
	omnibus.tree.base.tsv omnibus.tree.arrows.tsv omnibus.tree.anchors.tsv \
	omnibus.logistic.base.tsv omnibus.logistic.arrows.tsv omnibus.logistic.anchors.tsv \
	omnibus.forest.base.tsv omnibus.forest.arrows.tsv omnibus.forest.anchors.tsv
	python -u plot_interp.py \
		--input1 omnibus.logistic.base.tsv    omnibus.forest.base.tsv    omnibus.tree.base.tsv    \
		--input2 omnibus.logistic.arrows.tsv  omnibus.forest.arrows.tsv  omnibus.tree.arrows.tsv  \
		--input3 omnibus.logistic.anchors.tsv omnibus.forest.anchors.tsv omnibus.tree.anchors.tsv annots/omnibus.add_labels.tsv \
		--show --output plot_omnibus.pdf --bw

### Appendix D.1 (Details of Case Studies)

## More details of the experiments presented in Section 6.2. See Section 6.2 above.

### Appendix D.2 (Algorithm Runtime vs. Model Size)

## Figure 9

# Benchmark vs. model size for decision trees

BD2 := python -u bench_detect2.py data/adult/adult_cleaner.csv -s marital_status -c earning \
	-m decision-tree --seed 0 --nominal_encoding single --remove_sensitive --sub_sample 2000 \
	--association nmi

BD2df := data_bench_detect_adult_vs_models_tree.tsv

$(BD2df):
	echo "dataset_size\tmodel_size\tmodel_height\tsub_expressions\truntime1\truntime2" > $(BD2df)
	$(BD2) --max_depth 2   >> $(BD2df)
	$(BD2) --max_depth 3   >> $(BD2df)
	$(BD2) --max_depth 4   >> $(BD2df)
	$(BD2) --max_depth 5   >> $(BD2df)
	$(BD2) --max_depth 6   >> $(BD2df)
	$(BD2) --max_depth 7   >> $(BD2df)
	$(BD2) --max_depth 8   >> $(BD2df)
	$(BD2) --max_depth 9   >> $(BD2df)
	$(BD2) --max_depth 10  >> $(BD2df)
	$(BD2) --max_depth 11  >> $(BD2df)
	$(BD2) --max_depth 12  >> $(BD2df)
	$(BD2) --max_depth 13  >> $(BD2df)
	$(BD2) --max_depth 14  >> $(BD2df)

plot_bench_detect2: *.py $(BD2df)
	python -u plot_bench_detect2.py --input1 $(BD2df) \
		--show --output plot_bench_detect_vs_models_tree.pdf

# Benchmark vs. model size for random forests

BD3 := python -u bench_detect2.py data/adult/adult_cleaner.csv -s marital_status -c earning \
	-m random-forest --seed 0 --nominal_encoding single --remove_sensitive --sub_sample 2000 \
	--association nmi

BD3df := data_bench_detect_adult_vs_models_forest2.tsv

$(BD3df):
	echo "dataset_size\tmodel_size\tmodel_height\tsub_expressions\truntime1\truntime2" > $(BD3df)
	$(BD3) --forest_trees 3 --max_depth 1  >> $(BD3df)
	$(BD3) --forest_trees 3 --max_depth 2  >> $(BD3df)
	$(BD3) --forest_trees 3 --max_depth 3  >> $(BD3df)
	$(BD3) --forest_trees 3 --max_depth 4  >> $(BD3df)
	$(BD3) --forest_trees 3 --max_depth 5  >> $(BD3df)
	$(BD3) --forest_trees 3 --max_depth 6  >> $(BD3df)
	$(BD3) --forest_trees 3 --max_depth 7  >> $(BD3df)
	$(BD3) --forest_trees 3 --max_depth 8  >> $(BD3df)
	$(BD3) --forest_trees 3 --max_depth 9  >> $(BD3df)
	$(BD3) --forest_trees 3 --max_depth 10 >> $(BD3df)
	$(BD3) --forest_trees 3 --max_depth 11 >> $(BD3df)
	$(BD3) --forest_trees 3 --max_depth 12 >> $(BD3df)

plot_bench_detect3: *.py $(BD3df)
	python -u plot_bench_detect2.py --input1 $(BD3df) \
		--show --output plot_bench_detect_vs_models_forest.pdf

# Benchmark vs. model size for logistic regression

BD4 := python -u bench_detect2.py data/adult/adult_cleaner.csv -s marital_status -c earning \
	-m logistic --seed 0 --nominal_encoding single --remove_sensitive --sub_sample 2000 \
	--association nmi

BD4df := data_bench_detect_adult_vs_models_logistic.tsv

$(BD4df):
	echo "dataset_size\tmodel_size\tmodel_height\tsub_expressions\truntime1\truntime2" > $(BD4df)
	$(BD4) --reg_param 0.004  >> $(BD4df)
	$(BD4) --reg_param 0.0045 >> $(BD4df)
	$(BD4) --reg_param 0.005  >> $(BD4df)
	$(BD4) --reg_param 0.006  >> $(BD4df)
	$(BD4) --reg_param 0.0075 >> $(BD4df)
	$(BD4) --reg_param 0.008  >> $(BD4df)
	$(BD4) --reg_param 0.02   >> $(BD4df)
	$(BD4) --reg_param 0.03   >> $(BD4df)
	$(BD4) --reg_param 0.05   >> $(BD4df)
	$(BD4) --reg_param 0.07   >> $(BD4df)

plot_bench_detect4: *.py $(BD4df)
	@python -u plot_bench_detect2.py --input1 $(BD4df) \
		--show --output plot_bench_detect_vs_models_logistic.pdf

# Figure 9

plot_bench_model: *.py $(BD2df) $(BD3df) $(BD4df)
	@python -u plot_bench_detect2.py --input1 $(BD2df) $(BD3df) $(BD4df) \
		--show --output plot_bench_detect_vs_models_all.pdf
