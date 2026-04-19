#!/bin/bash
# Group 1
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 10000 --sparsity 0.002 --noise 0.2
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 10000 --sparsity 0.002 --noise 0.2
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 10000 --sparsity 0.002 --noise 0.2 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 10000 --sparsity 0.002 --noise 0.2 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 10000 --sparsity 0.002 --noise 0.2
python generate/generate_model.py --topology random --n_users 2000 --n_items 10000 --sparsity 0.002 --noise 0.2
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 10000 --sparsity 0.002 --noise 0.2
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 10000 --sparsity 0.002 --noise 0.2
python generate/generate_topology.py --topology star --n_users 2000 --n_items 10000 --sparsity 0.002 --noise 0.2

# Group 2
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.2
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.2
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.2 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.2 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.2
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.2
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.2
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.2
python generate/generate_topology.py --topology star --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.2

# Group 3
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_topology.py --topology star --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1

# Group 4
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.1
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.1
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.1 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.1 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.1
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.1
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.1
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.1
python generate/generate_topology.py --topology star --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.1

# Group 5
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.005 --noise 0.1
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.005 --noise 0.1
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 5000 --sparsity 0.005 --noise 0.1 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 5000 --sparsity 0.005 --noise 0.1 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.005 --noise 0.1
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.005 --noise 0.1
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 5000 --sparsity 0.005 --noise 0.1
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 5000 --sparsity 0.005 --noise 0.1
python generate/generate_topology.py --topology star --n_users 2000 --n_items 5000 --sparsity 0.005 --noise 0.1

# Group 6
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.2
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.2
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.2 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.2 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.2
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.2
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.2
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.2
python generate/generate_topology.py --topology star --n_users 2000 --n_items 5000 --sparsity 0.02 --noise 0.2

# Group 7
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.05
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.05
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.05 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.05 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.05
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.05
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.05
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.05
python generate/generate_topology.py --topology star --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.05

# Group 8
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 2000 --sparsity 0.01 --noise 0.1
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 2000 --sparsity 0.01 --noise 0.1
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 2000 --sparsity 0.01 --noise 0.1 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 2000 --sparsity 0.01 --noise 0.1 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 2000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology random --n_users 2000 --n_items 2000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 2000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 2000 --sparsity 0.01 --noise 0.1
python generate/generate_topology.py --topology star --n_users 2000 --n_items 2000 --sparsity 0.01 --noise 0.1

# Group 9
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 2000 --sparsity 0.02 --noise 0.1
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 2000 --sparsity 0.02 --noise 0.1
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 2000 --sparsity 0.02 --noise 0.1 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 2000 --sparsity 0.02 --noise 0.1 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 2000 --sparsity 0.02 --noise 0.1
python generate/generate_model.py --topology random --n_users 2000 --n_items 2000 --sparsity 0.02 --noise 0.1
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 2000 --sparsity 0.02 --noise 0.1
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 2000 --sparsity 0.02 --noise 0.1
python generate/generate_topology.py --topology star --n_users 2000 --n_items 2000 --sparsity 0.02 --noise 0.1

# Group 10
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 10000 --sparsity 0.01 --noise 0.1
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 10000 --sparsity 0.01 --noise 0.1
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 10000 --sparsity 0.01 --noise 0.1 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 10000 --sparsity 0.01 --noise 0.1 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 10000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology random --n_users 2000 --n_items 10000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 10000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 10000 --sparsity 0.01 --noise 0.1
python generate/generate_topology.py --topology star --n_users 2000 --n_items 10000 --sparsity 0.01 --noise 0.1

# Group 11
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.3
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.3
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.3 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.3 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.3
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.3
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.3
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.3
python generate/generate_topology.py --topology star --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.3

# Group 12
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.5
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.5
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.5 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.5 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.5
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.5
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.5
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.5
python generate/generate_topology.py --topology star --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.5

# Group 13
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.02
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.02
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.02 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.02 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.02
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.02
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.02
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.02
python generate/generate_topology.py --topology star --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.02

# Group 14
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.003 --noise 0.1
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.003 --noise 0.1
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 5000 --sparsity 0.003 --noise 0.1 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 5000 --sparsity 0.003 --noise 0.1 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.003 --noise 0.1
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.003 --noise 0.1
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 5000 --sparsity 0.003 --noise 0.1
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 5000 --sparsity 0.003 --noise 0.1
python generate/generate_topology.py --topology star --n_users 2000 --n_items 5000 --sparsity 0.003 --noise 0.1

# Group 15
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.05 --noise 0.1
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.05 --noise 0.1
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 5000 --sparsity 0.05 --noise 0.1 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 2000 --n_items 5000 --sparsity 0.05 --noise 0.1 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.05 --noise 0.1
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.05 --noise 0.1
python generate/generate_model.py --topology scale_free --n_users 2000 --n_items 5000 --sparsity 0.05 --noise 0.1
python generate/generate_model.py --topology small_world --n_users 2000 --n_items 5000 --sparsity 0.05 --noise 0.1
python generate/generate_topology.py --topology star --n_users 2000 --n_items 5000 --sparsity 0.05 --noise 0.1

# Groups 16-18: fake_ratio ablation
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --fake_ratio 0.05
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --fake_ratio 0.2
python generate/generate_topology.py --topology fake_users --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --fake_ratio 0.4

# Groups 19-22: shared_dims ablation
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --shared_dims 1
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --shared_dims 3
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --shared_dims 7
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --shared_dims 9

# Groups 23-26: avg_degree ablation
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --er_p 0.001
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --avg_degree 2.0
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --avg_degree 2.0
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --avg_degree 2.0
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --er_p 0.002
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --avg_degree 4.0
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --avg_degree 4.0
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --avg_degree 4.0
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --er_p 0.005
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --avg_degree 10.0
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --avg_degree 10.0
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --avg_degree 10.0
python generate/generate_model.py --topology random --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --er_p 0.01
python generate/generate_social.py --topology echo_chamber --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --avg_degree 20.0
python generate/generate_social.py --topology partial_alignment --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --avg_degree 20.0
python generate/generate_social.py --topology contrarian --n_users 2000 --n_items 5000 --sparsity 0.01 --noise 0.1 --avg_degree 20.0

# Group 27: u500
python generate/generate_social.py --topology contrarian --n_users 500 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_social.py --topology echo_chamber --n_users 500 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_topology.py --topology fake_users --n_users 500 --n_items 5000 --sparsity 0.01 --noise 0.1 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 500 --n_items 5000 --sparsity 0.01 --noise 0.1 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 500 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology random --n_users 500 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology scale_free --n_users 500 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology small_world --n_users 500 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_topology.py --topology star --n_users 500 --n_items 5000 --sparsity 0.01 --noise 0.1

# Group 28: u5000
python generate/generate_social.py --topology contrarian --n_users 5000 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_social.py --topology echo_chamber --n_users 5000 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_topology.py --topology fake_users --n_users 5000 --n_items 5000 --sparsity 0.01 --noise 0.1 --fake_ratio 0.1
python generate/generate_topology.py --topology line_graph --n_users 5000 --n_items 5000 --sparsity 0.01 --noise 0.1 --max_interactions 30 --decay 0.5
python generate/generate_social.py --topology partial_alignment --n_users 5000 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology random --n_users 5000 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology scale_free --n_users 5000 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_model.py --topology small_world --n_users 5000 --n_items 5000 --sparsity 0.01 --noise 0.1
python generate/generate_topology.py --topology star --n_users 5000 --n_items 5000 --sparsity 0.01 --noise 0.1