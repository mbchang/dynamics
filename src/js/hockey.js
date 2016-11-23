var Matter = require('matter-js')
var jsonfile = require('jsonfile')
var _ = require('underscore')

// Define everything within the scope of this file
module.exports = {

function () {

    // module aliases
    var Engine = Matter.Engine,
        World = Matter.World,
        Bodies = Matter.Bodies,
        Body = Matter.Body;

    var Hockey = {}

    Hockey.create = function(env) {
        var self = {}; // instance of the Hockey class

        // these should not be mutated
        var params = {show: false,
                      num_obj: 3,
                      cx: 400,  // 640
                      cy: 300,  // 480
                      max_v0: 20,
                      obj_radius: 50 };
        self.params = params;

        // var engine = Engine.create();
        var engine = env.engine
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
                                                         inverseInertia: 0}));

             // set velocities
             _.each(_.zip(self.engine.world.bodies
                         .filter(function(elem) {
                                     return elem.label === 'Circle Body';
                                 }), self.v0),
                         function(pair){
                             Body.setVelocity(pair[0],pair[1]); // pair[1] is object, pair[2] is velocity
                         });
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

    return Hockey;
}
}
