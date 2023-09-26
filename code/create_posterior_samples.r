library(ape)
library(tidyverse)

# Load the trees


datasets <- 
    list.files("mrbayes/output", pattern = "run1.t$", full.names = TRUE) %>%
    split(., basename(.)) %>%
    names %>%
    split(., sub(".run1.t.*", "", .)) %>%
    names

dir.create("../data/posterior_trees", showWarnings = FALSE)

for (ds in datasets) {
    if (ds != "houchinese_correspondences") {
        print(ds)
        tree_list <- 
            list(
                read.nexus(paste0("mrbayes/output/", ds, ".run1.t")),
                read.nexus(paste0("mrbayes/output/", ds, ".run2.t")),
                read.nexus(paste0("mrbayes/output/", ds, ".run3.t")),
                read.nexus(paste0("mrbayes/output/", ds, ".run4.t"))
            ) 
        trees <- NULL
        for (i in 1:length(tree_list)) {
            trees_i <- tree_list[[i]]
            bi <- ceiling(length(trees_i)/4)
            trees_i <- trees_i[(bi+1):length(trees_i)]
            if (i == 1) {
                trees <- trees_i
                } else {
            trees <- c(trees, trees_i)
                }
        }
        trees <- sample(trees, 1000)
        write.tree(trees, file = paste0("../data/posterior_trees/", ds, ".trees"))
    }
}
