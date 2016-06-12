var Matter = require('matter-js')
var jsonfile = require('jsonfile')
var _ = require('underscore')
require('./demo/js/Examples')

// module aliases
var Engine = Matter.Engine,
    World = Matter.World,
    Bodies = Matter.Bodies,
    Body = Matter.Body,
    Example = Matter.Example;


// some example engine options
var engine_options = {
    positionIterations: 6,
    velocityIterations: 4,
    enableSleeping: false,
    metrics: { extended: true }
};

var env = {}
env.engine = Engine.create(engine_options);


var scenarios = {
    balls: "m_balls",
    cradle: "m_cradle",
    tower: "m_tower"
}


// ultimately this is the format you want:
// (num_samples x max_other_objects x num_past x 8)

// so I still need to figure out if simulate will do multiple examples or not,
// but assuming that simulate does one example, then the trajectory level would
// // be: obj --> step --> [state]

simulate = function(scenario, numsteps) {
    var sim_file = 'out.json',  // TODO: eventually this will contain the scenario name
        trajectory = [],
        i, id, k;

    // initialize trajectory conatiner
    for (id = 0; id < scenario.params.num_obj; id++) { //id = 0 corresponds to world!
        trajectory[id] = [];
    }

    // run the engine
    for (i = 0; i < numsteps; i++) {
        for (id = 0; id < scenario.params.num_obj; id++) { //id = 0 corresponds to world!
            // Now it is a design choice of whether to use keys or to use numbers
            trajectory[id][i] = {};
            for (k of ['position', 'velocity', 'mass']){
                if (typeof scenario.engine.world.bodies[id][k] === 'object') {
                    trajectory[id][i][k] = Matter.Common.clone(scenario.engine.world.bodies[id][k], true); // can only clone objects, not primitives
                } else {
                    trajectory[id][i][k] = scenario.engine.world.bodies[id][k]; // can only clone objects, not primitives
                }
            }
        }
        Engine.update(scenario.engine);
    }
    jsonfile.writeFileSync(sim_file, trajectory, {spaces: 2});
};

simulate(Example[scenarios.balls](env), 5);
