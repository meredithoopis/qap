"""QAP: copy / substitute / abstain via a robot reachability quasimetric.

Research code built on top of The Imitator Game (IG) repository. Week 1 and Week 2
of the execution plan live here:

    qap.common      task table, environment factory, episode I/O
    qap.state       object-centric state extractor, contact-onset (entry state) recorder
    qap.week1       task inventory, screening rule, frozen splits, shortcut filter
    qap.collect     state-recording of IG motion-planning episodes, robot play generator
    qap.quasimetric d_theta: object-token transformer + IQE/MRN/Hilbert heads, QRL/TMD/HILP
    qap.feasibility planner-certified feasibility labels, PR-AUC of distances
    qap.policy      AWR goal-conditioned policy, oracle-goal reaching evaluation
    qap.scenes      graded IG-Sub scenes G0-G4, episode generation, planner feasibility

See qap/RUN_GUIDE.md for the commands.
"""

__version__ = "0.1.0"
