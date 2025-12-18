namespace pomdp {

class Loop {
public:
    Loop(BeliefUpdater&, Policy&);
    void step(Observation&);
};

}