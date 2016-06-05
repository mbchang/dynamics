var Matter = require('matter-js')
var jsonfile = require('jsonfile')

// common functions
euc_dist = function(p1, p2) {
    var x2 = Math.pow(p1.x - p2.x, 2),
        y2 = Math.pow(p1.y - p2.y, 2);
    return Math.sqrt(x2 + y2);
};

rand_pos = function(x_bounds, y_bounds) {

    var xrange = x_bounds.hi - x_bounds.lo,
        yrange = y_bounds.hi - y_bounds.lo;
    var px = Math.floor((Math.random()*(xrange))) + x_bounds.lo,
        py = Math.floor((Math.random()*(yrange))) + y_bounds.lo;
    return {x: px, y: py};
};

// Define everything within the scope of this file
(function () {

// module aliases
var Engine = Matter.Engine,
    World = Matter.World,
    Bodies = Matter.Bodies;

// I could have something like RBD.create() (Rigid Body Dynamics)
var Hockey = {}

Hockey.create = function(options) {
    var self = {}; // instance of the Hockey class

    // these should not be mutated
    var params = {show: false,
                  num_obj: 3,
                  cx: 400,
                  cy: 300,
                  max_v0: 20,
                  obj_radius: 50 };
    self.params = params;

    var engine = Engine.create();
    engine.world.gravity.y = 0;
    engine.world.gravity.x = 0;
    engine.world.bounds = { min: { x: 0, y: 0 },
                            max: { x: 2*params.cx, y: 2*params.cy }}
    self.engine = engine;

    // function
    self.rand_pos = function() {
        return rand_pos(
            {hi: 2*params.cx - params.obj_radius - 1, lo: params.obj_radius + 1},
            {hi: 2*params.cy - params.obj_radius - 1, lo: params.obj_radius + 1});
        };

    return self
};


Hockey.init = function(self) {  // hockey is like self here
    var i; // iterator

    self.v0 = [];  // initial velocities
    self.p0 = [];  // initial positions

    for (i = 0; i < self.params.num_obj; i++) {

        // generate random initial velocities b/w 0 and max_v0 inclusive
        var vx = Math.floor(Math.random()*self.params.max_v0+1)
        var vy = Math.floor(Math.random()*self.params.max_v0+1)
        self.v0.push({x: vx, y: vy})

        // generate random initial positions by rejection sampling
        if (self.p0.length == 0) {  // assume that num_obj > 0
            self.p0.push(self.rand_pos());
        } else {
            var proposed_pos = self.rand_pos();
            // true if overlaps
            while ((function(){
                    for (var j = 0; j < self.p0.length; j++) {
                        if (euc_dist(proposed_pos, self.p0[j]) < 1.5*self.params.obj_radius) {
                            return true;
                        }
                    }
                    return false;
                })()){
                // keep trying until you get a match
                proposed_pos = self.rand_pos();
            }
            self.p0.push(proposed_pos);
        }

        // add body to world
        World.add(self.engine.world,
            Bodies.circle(self.p0[i].x, self.p0[i].y, self.params.obj_radius,
                                                    {restitution: 1,
                                                     friction: 0,
                                                     frictionAir: 0,
                                                     frictionStatic: 0,
                                                     inertia: Infinity,
                                                     inverseInertia: 0,
                                                     velocity: self.v0[i]}));
    }

    // world boundaries. This is okay because we the circles don't depend on the boundaries
    var offset = 5;  // world offset
    World.add(self.engine.world, [
        Bodies.rectangle(self.params.cx, -offset, 2*self.params.cx + 2*offset, 2*offset, { isStatic: true, restitution: 1 }),
        Bodies.rectangle(self.params.cx, 600+offset, 2*self.params.cx + 2*offset, 2*offset, { isStatic: true, restitution: 1 }),
        Bodies.rectangle(2*self.params.cx + offset, self.params.cy, 2*offset, 2*self.params.cy + 2*offset, { isStatic: true, restitution: 1 }),
        Bodies.rectangle(-offset, self.params.cy, 2*offset, 2*self.params.cy + 2*offset, { isStatic: true, restitution: 1 })
    ]);
};


// Eventually I have to decouple this out of the scope of the Hockey definition
// main
var hockey = Hockey.create()
Hockey.init(hockey)
// console.log(Matter.Common.values(hockey.engine.world.bodies[0].velocity))



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
        // TODO: why is it the case that when the engine updates, the velocities change to 0?
        // This could be a design choice, but I want the velocities to persist! Given initial velocities, I want them to keep going!
        // I can do this on Demo and ask about it.
    }
    jsonfile.writeFileSync(sim_file, trajectory);
};

simulate(hockey, 5);


})();



// note that Hockey is outside the scope now!

// So for all scenarios, you just do
// var scenario = Scenario.create()
// Scenario.init(scenario)
// simulate(scenario, numsteps)

// this is outside the Hockey scope!
// // TODO: in the "main" scope, should I make Engine, World, Composites, etc global?
