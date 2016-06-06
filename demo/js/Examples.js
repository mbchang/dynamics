var _isBrowser = typeof window !== 'undefined' && window.location,
    // Matter = _isBrowser ? window.Matter : require('../../build/matter-dev.js');
    Matter = _isBrowser ? window.Matter : require('matter-js');  // how do I modify window.Matter? Also it looks like I need to download matter-dev.js!

var Example = {};
Matter.Example = Example;

if (!_isBrowser) {
    module.exports = Example;
}

(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies;

    Example.airFriction = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        World.add(world, [
            Bodies.rectangle(200, 100, 60, 60, { frictionAir: 0.001 }),
            Bodies.rectangle(400, 100, 60, 60, { frictionAir: 0.05 }),
            Bodies.rectangle(600, 100, 60, 60, { frictionAir: 0.1 })
        ]);

        var renderOptions = demo.render.options;
        renderOptions.showAngleIndicator = false;
        renderOptions.showVelocity = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Common = Matter.Common;

    Example.avalanche = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var stack = Composites.stack(20, 20, 20, 5, 0, 0, function(x, y) {
            return Bodies.circle(x, y, Common.random(10, 20), { friction: 0.00001, restitution: 0.5, density: 0.001 });
        });

        World.add(world, stack);

        World.add(world, [
            Bodies.rectangle(200, 150, 700, 20, { isStatic: true, angle: Math.PI * 0.06 }),
            Bodies.rectangle(500, 350, 700, 20, { isStatic: true, angle: -Math.PI * 0.06 }),
            Bodies.rectangle(340, 580, 700, 20, { isStatic: true, angle: Math.PI * 0.04 })
        ]);
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Common = Matter.Common;

    Example.ballPool = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var stack = Composites.stack(100, 50, 10, 15, 10, 10, function(x, y) {
            return Bodies.circle(x, y, Common.random(15, 30), { restitution: 0.6, friction: 0.1 });
        });

        World.add(world, [
            stack,
            Bodies.polygon(200, 560, 3, 60),
            Bodies.polygon(400, 560, 5, 60),
            Bodies.rectangle(600, 560, 80, 80)
        ]);
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites;

    Example.beachBalls = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var stack = Composites.stack(0, 100, 5, 1, 20, 0, function(x, y) {
            return Bodies.circle(x, y, 75, { restitution: 0.9 });
        });

        World.add(world, stack);
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Body = Matter.Body,
        Composites = Matter.Composites,
        Common = Matter.Common,
        Constraint = Matter.Constraint;

    Example.bridge = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            group = Body.nextGroup(true);

        var bridge = Composites.stack(150, 300, 9, 1, 10, 10, function(x, y) {
            return Bodies.rectangle(x, y, 50, 20, { collisionFilter: { group: group } });
        });

        Composites.chain(bridge, 0.5, 0, -0.5, 0, { stiffness: 0.9 });

        var stack = Composites.stack(200, 40, 6, 3, 0, 0, function(x, y) {
            return Bodies.polygon(x, y, Math.round(Common.random(1, 8)), Common.random(20, 40));
        });

        World.add(world, [
            bridge,
            Bodies.rectangle(80, 440, 120, 280, { isStatic: true }),
            Bodies.rectangle(720, 440, 120, 280, { isStatic: true }),
            Constraint.create({ pointA: { x: 140, y: 300 }, bodyB: bridge.bodies[0], pointB: { x: -25, y: 0 } }),
            Constraint.create({ pointA: { x: 660, y: 300 }, bodyB: bridge.bodies[8], pointB: { x: 25, y: 0 } }),
            stack
        ]);
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Common = Matter.Common;

    Example.broadphase = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var stack = Composites.stack(20, 20, 20, 5, 0, 0, function(x, y) {
            switch (Math.round(Common.random(0, 1))) {

            case 0:
                if (Common.random() < 0.8) {
                    return Bodies.rectangle(x, y, Common.random(20, 50), Common.random(20, 50));
                } else {
                    return Bodies.rectangle(x, y, Common.random(80, 120), Common.random(20, 30));
                }
                break;
            case 1:
                return Bodies.polygon(x, y, Math.round(Common.random(1, 8)), Common.random(20, 50));

            }
        });

        World.add(world, stack);

        var renderOptions = demo.render.options;
        renderOptions.showBroadphase = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites;

    Example.car = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            scale;

        scale = 0.9;
        World.add(world, Composites.car(150, 100, 100 * scale, 40 * scale, 30 * scale));

        scale = 0.8;
        World.add(world, Composites.car(350, 300, 100 * scale, 40 * scale, 30 * scale));

        World.add(world, [
            Bodies.rectangle(200, 150, 650, 20, { isStatic: true, angle: Math.PI * 0.06 }),
            Bodies.rectangle(500, 350, 650, 20, { isStatic: true, angle: -Math.PI * 0.06 }),
            Bodies.rectangle(340, 580, 700, 20, { isStatic: true, angle: Math.PI * 0.04 })
        ]);

        var renderOptions = demo.render.options;
        renderOptions.showAngleIndicator = true;
        renderOptions.showCollisions = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Constraint = Matter.Constraint;

    Example.catapult = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var stack = Composites.stack(250, 255, 1, 6, 0, 0, function(x, y) {
            return Bodies.rectangle(x, y, 30, 30);
        });

        var catapult = Bodies.rectangle(400, 520, 320, 20, {  });

        World.add(world, [
            stack,
            catapult,
            Bodies.rectangle(250, 555, 20, 50, { isStatic: true }),
            Bodies.circle(560, 100, 50, { density: 0.005 }),
            Constraint.create({ bodyA: catapult, pointB: { x: 390, y: 580 } }),
            Constraint.create({ bodyA: catapult, pointB: { x: 410, y: 580 } })
        ]);

        var renderOptions = demo.render.options;
        renderOptions.showCollisions = true;
        renderOptions.showVelocity = true;
        renderOptions.showAngleIndicator = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Body = Matter.Body,
        Composite = Matter.Composite,
        Composites = Matter.Composites,
        Constraint = Matter.Constraint;

    Example.chains = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            group = Body.nextGroup(true);

        var ropeA = Composites.stack(200, 100, 5, 2, 10, 10, function(x, y) {
            return Bodies.rectangle(x, y, 50, 20, { collisionFilter: { group: group } });
        });

        Composites.chain(ropeA, 0.5, 0, -0.5, 0, { stiffness: 0.8, length: 2 });
        Composite.add(ropeA, Constraint.create({
            bodyB: ropeA.bodies[0],
            pointB: { x: -25, y: 0 },
            pointA: { x: 200, y: 100 },
            stiffness: 0.5
        }));

        World.add(world, ropeA);

        group = Body.nextGroup(true);

        var ropeB = Composites.stack(500, 100, 5, 2, 10, 10, function(x, y) {
            return Bodies.circle(x, y, 20, { collisionFilter: { group: group } });
        });

        Composites.chain(ropeB, 0.5, 0, -0.5, 0, { stiffness: 0.8, length: 2 });
        Composite.add(ropeB, Constraint.create({
            bodyB: ropeB.bodies[0],
            pointB: { x: -20, y: 0 },
            pointA: { x: 500, y: 100 },
            stiffness: 0.5
        }));

        World.add(world, ropeB);
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites;

    Example.circleStack = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var stack = Composites.stack(100, 185, 10, 10, 20, 0, function(x, y) {
            return Bodies.circle(x, y, 20);
        });

        World.add(world, stack);
    };

})();

(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Body = Matter.Body,
        Composites = Matter.Composites;

    Example.cloth = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var group = Body.nextGroup(true),
            particleOptions = { friction: 0.00001, collisionFilter: { group: group }, render: { visible: false }},
            cloth = Composites.softBody(200, 200, 20, 12, 5, 5, false, 8, particleOptions);

        for (var i = 0; i < 20; i++) {
            cloth.bodies[i].isStatic = true;
        }

        World.add(world, [
            cloth,
            Bodies.circle(300, 500, 80, { isStatic: true }),
            Bodies.rectangle(500, 480, 80, 80, { isStatic: true })
        ]);
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Common = Matter.Common;

    Example.collisionFiltering = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            mouseConstraint = demo.mouseConstraint;

        // define our categories (as bit fields, there are up to 32 available)
        var defaultCategory = 0x0001,
            redCategory = 0x0002,
            greenCategory = 0x0004,
            blueCategory = 0x0008;

        var redColor = '#C44D58',
            blueColor = '#4ECDC4',
            greenColor = '#C7F464';

        // create a stack with varying body categories (but these bodies can all collide with each other)
        World.add(world,
            Composites.stack(275, 150, 5, 10, 10, 10, function(x, y, column, row) {
                var category = redCategory,
                    color = redColor;

                if (row > 5) {
                    category = blueCategory;
                    color = blueColor;
                } else if (row > 2) {
                    category = greenCategory;
                    color = greenColor;
                }

                return Bodies.circle(x, y, 20, {
                    collisionFilter: {
                        category: category
                    },
                    render: {
                        strokeStyle: color,
                        fillStyle: 'transparent'
                    }
                });
            })
        );

        // this body will only collide with the walls and the green bodies
        World.add(world,
            Bodies.circle(310, 40, 30, {
                collisionFilter: {
                    mask: defaultCategory | greenCategory
                },
                render: {
                    strokeStyle: Common.shadeColor(greenColor, -20),
                    fillStyle: greenColor
                }
            })
        );

        // this body will only collide with the walls and the red bodies
        World.add(world,
            Bodies.circle(400, 40, 30, {
                collisionFilter: {
                    mask: defaultCategory | redCategory
                },
                render: {
                    strokeStyle: Common.shadeColor(redColor, -20),
                    fillStyle: redColor
                }
            })
        );

        // this body will only collide with the walls and the blue bodies
        World.add(world,
            Bodies.circle(480, 40, 30, {
                collisionFilter: {
                    mask: defaultCategory | blueCategory
                },
                render: {
                    strokeStyle: Common.shadeColor(blueColor, -20),
                    fillStyle: blueColor
                }
            })
        );

        // red category objects should not be draggable with the mouse
        mouseConstraint.collisionFilter.mask = defaultCategory | blueCategory | greenCategory;

        var renderOptions = demo.render.options;
        renderOptions.wireframes = false;
        renderOptions.background = '#222';
        renderOptions.showAngleIndicator = false;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composite = Matter.Composite,
        Composites = Matter.Composites,
        Events = Matter.Events;

    Example.compositeManipulation = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            sceneEvents = demo.sceneEvents;

        var stack = Composites.stack(200, 200, 4, 4, 0, 0, function(x, y) {
            return Bodies.rectangle(x, y, 40, 40);
        });

        World.add(world, stack);

        world.gravity.y = 0;

        sceneEvents.push(
            Events.on(engine, 'afterUpdate', function(event) {
                var time = engine.timing.timestamp;

                Composite.translate(stack, {
                    x: Math.sin(time * 0.001) * 2,
                    y: 0
                });

                Composite.rotate(stack, Math.sin(time * 0.001) * 0.01, {
                    x: 300,
                    y: 300
                });

                var scale = 1 + (Math.sin(time * 0.001) * 0.01);

                Composite.scale(stack, scale, scale, {
                    x: 300,
                    y: 300
                });
            })
        );

        var renderOptions = demo.render.options;
        renderOptions.wireframes = false;
        renderOptions.showAxes = true;
        renderOptions.showCollisions = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Body = Matter.Body,
        Constraint = Matter.Constraint;

    Example.compound = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var size = 200,
            x = 200,
            y = 200,
            partA = Bodies.rectangle(x, y, size, size / 5),
            partB = Bodies.rectangle(x, y, size / 5, size, { render: partA.render });

        var compoundBodyA = Body.create({
            parts: [partA, partB]
        });

        size = 150;
        x = 400;
        y = 300;

        var partC = Bodies.circle(x, y, 30),
            partD = Bodies.circle(x + size, y, 30),
            partE = Bodies.circle(x + size, y + size, 30),
            partF = Bodies.circle(x, y + size, 30);

        var compoundBodyB = Body.create({
            parts: [partC, partD, partE, partF]
        });

        var constraint = Constraint.create({
            pointA: { x: 400, y: 100 },
            bodyB: compoundBodyB,
            pointB: { x: 0, y: -50 }
        });

        World.add(world, [compoundBodyA, compoundBodyB, constraint]);

        var renderOptions = demo.render.options;
        renderOptions.showAxes = true;
        renderOptions.showPositions = true;
        renderOptions.showConvexHulls = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Body = Matter.Body,
        Composites = Matter.Composites;

    Example.compoundStack = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            size = 50;

        var stack = Composites.stack(100, 280, 12, 6, 0, 0, function(x, y) {
            var partA = Bodies.rectangle(x, y, size, size / 5),
                partB = Bodies.rectangle(x, y, size / 5, size, { render: partA.render });

            return Body.create({
                parts: [partA, partB]
            });
        });

        World.add(world, stack);
    };

})();

(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Common = Matter.Common,
        Vertices = Matter.Vertices;

    Example.concave = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var arrow = Vertices.fromPath('40 0 40 20 100 20 100 80 40 80 40 100 0 50'),
            chevron = Vertices.fromPath('100 0 75 50 100 100 25 100 0 50 25 0'),
            star = Vertices.fromPath('50 0 63 38 100 38 69 59 82 100 50 75 18 100 31 59 0 38 37 38'),
            horseShoe = Vertices.fromPath('35 7 19 17 14 38 14 58 25 79 45 85 65 84 65 66 46 67 34 59 30 44 33 29 45 23 66 23 66 7 53 7');

        var stack = Composites.stack(50, 50, 6, 4, 10, 10, function(x, y) {
            var color = Common.choose(['#556270', '#4ECDC4', '#C7F464', '#FF6B6B', '#C44D58']);
            return Bodies.fromVertices(x, y, Common.choose([arrow, chevron, star, horseShoe]), {
                render: {
                    fillStyle: color,
                    strokeStyle: color
                }
            }, true);
        });

        World.add(world, stack);

        var renderOptions = demo.render.options;
        renderOptions.showAngleIndicator = false;
    };

})();
(function() {

    var Engine = Matter.Engine;

    Example.engine = function(demo) {
        // some example engine options
        var options = {
            positionIterations: 6,
            velocityIterations: 4,
            enableSleeping: false,
            metrics: { extended: true }
        };

        return Engine.create(options);
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Body = Matter.Body,
        Composite = Matter.Composite,
        Composites = Matter.Composites,
        Common = Matter.Common,
        Events = Matter.Events;

    Example.events = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            mouseConstraint = demo.mouseConstraint,
            sceneEvents = demo.sceneEvents;

        // bind events (sceneEvents is only used for this demo)

        sceneEvents.push(

            // an example of using composite events on the world
            Events.on(world, 'afterAdd', function(event) {
                console.log('added to world:', event.object);
            })

        );

        sceneEvents.push(

            // an example of using beforeUpdate event on an engine
            Events.on(engine, 'beforeUpdate', function(event) {
                var engine = event.source;

                // apply random forces every 5 secs
                if (event.timestamp % 5000 < 50)
                    shakeScene(engine);
            })

        );

        sceneEvents.push(

            // an example of using collisionStart event on an engine
            Events.on(engine, 'collisionStart', function(event) {
                var pairs = event.pairs;

                // change object colours to show those starting a collision
                for (var i = 0; i < pairs.length; i++) {
                    var pair = pairs[i];
                    pair.bodyA.render.fillStyle = '#bbbbbb';
                    pair.bodyB.render.fillStyle = '#bbbbbb';
                }
            })

        );

        sceneEvents.push(

            // an example of using collisionActive event on an engine
            Events.on(engine, 'collisionActive', function(event) {
                var pairs = event.pairs;

                // change object colours to show those in an active collision (e.g. resting contact)
                for (var i = 0; i < pairs.length; i++) {
                    var pair = pairs[i];
                    pair.bodyA.render.fillStyle = '#aaaaaa';
                    pair.bodyB.render.fillStyle = '#aaaaaa';
                }
            })

        );

        sceneEvents.push(

            // an example of using collisionEnd event on an engine
            Events.on(engine, 'collisionEnd', function(event) {
                var pairs = event.pairs;

                // change object colours to show those ending a collision
                for (var i = 0; i < pairs.length; i++) {
                    var pair = pairs[i];
                    pair.bodyA.render.fillStyle = '#999999';
                    pair.bodyB.render.fillStyle = '#999999';
                }
            })

        );

        sceneEvents.push(

            // an example of using mouse events on a mouse
            Events.on(mouseConstraint, 'mousedown', function(event) {
                var mousePosition = event.mouse.position;
                console.log('mousedown at ' + mousePosition.x + ' ' + mousePosition.y);
                demo.render.options.background = 'cornsilk';
                shakeScene(engine);
            })

        );

        sceneEvents.push(

            // an example of using mouse events on a mouse
            Events.on(mouseConstraint, 'mouseup', function(event) {
                var mousePosition = event.mouse.position;
                demo.render.options.background = "white";
                console.log('mouseup at ' + mousePosition.x + ' ' + mousePosition.y);
            })

        );

        sceneEvents.push(

            // an example of using mouse events on a mouse
            Events.on(mouseConstraint, 'startdrag', function(event) {
                console.log('startdrag', event);
            })

        );

        sceneEvents.push(

            // an example of using mouse events on a mouse
            Events.on(mouseConstraint, 'enddrag', function(event) {
                console.log('enddrag', event);
            })

        );

        // scene code

        var stack = Composites.stack(50, 100, 8, 4, 50, 50, function(x, y) {
            return Bodies.circle(x, y, 15, { restitution: 1, render: { strokeStyle: '#777' } });
        });

        World.add(world, stack);

        var renderOptions = demo.render.options;
        renderOptions.wireframes = false;

        var shakeScene = function(engine) {
            var bodies = Composite.allBodies(engine.world);

            for (var i = 0; i < bodies.length; i++) {
                var body = bodies[i];

                if (!body.isStatic && body.position.y >= 500) {
                    var forceMagnitude = 0.01 * body.mass;

                    Body.applyForce(body, { x: 0, y: 0 }, {
                        x: (forceMagnitude + Common.random() * forceMagnitude) * Common.choose([1, -1]),
                        y: -forceMagnitude + Common.random() * -forceMagnitude
                    });
                }
            }
        };
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies;

    Example.friction = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        World.add(world, [
            Bodies.rectangle(300, 180, 700, 20, { isStatic: true, angle: Math.PI * 0.06 }),
            Bodies.rectangle(300, 70, 40, 40, { friction: 0.001 })
        ]);

        World.add(world, [
            Bodies.rectangle(300, 350, 700, 20, { isStatic: true, angle: Math.PI * 0.06 }),
            Bodies.rectangle(300, 250, 40, 40, { friction: 0.0005 })
        ]);

        World.add(world, [
            Bodies.rectangle(300, 520, 700, 20, { isStatic: true, angle: Math.PI * 0.06 }),
            Bodies.rectangle(300, 430, 40, 40, { friction: 0 })
        ]);

        var renderOptions = demo.render.options;
        renderOptions.showAngleIndicator = false;
        renderOptions.showVelocity = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Common = Matter.Common;

    Example.gravity = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        engine.world.gravity.y = -1;

        var stack = Composites.stack(20, 20, 20, 5, 0, 0, function(x, y) {
            switch (Math.round(Common.random(0, 1))) {

            case 0:
                if (Common.random() < 0.8) {
                    return Bodies.rectangle(x, y, Common.random(20, 50), Common.random(20, 50));
                } else {
                    return Bodies.rectangle(x, y, Common.random(80, 120), Common.random(20, 30));
                }
                break;
            case 1:
                return Bodies.polygon(x, y, Math.round(Common.random(1, 8)), Common.random(20, 50));

            }
        });

        World.add(world, stack);
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Body = Matter.Body,
        Events = Matter.Events;

    Example.manipulation = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            sceneEvents = demo.sceneEvents;

        var bodyA = Bodies.rectangle(100, 200, 50, 50, { isStatic: true }),
            bodyB = Bodies.rectangle(200, 200, 50, 50),
            bodyC = Bodies.rectangle(300, 200, 50, 50),
            bodyD = Bodies.rectangle(400, 200, 50, 50),
            bodyE = Bodies.rectangle(550, 200, 50, 50),
            bodyF = Bodies.rectangle(700, 200, 50, 50),
            bodyG = Bodies.circle(400, 100, 25),
            partA = Bodies.rectangle(600, 200, 120, 50),
            partB = Bodies.rectangle(660, 200, 50, 190),
            compound = Body.create({
                parts: [partA, partB],
                isStatic: true
            });

        World.add(world, [bodyA, bodyB, bodyC, bodyD, bodyE, bodyF, bodyG, compound]);

        var counter = 0,
            scaleFactor = 1.01;

        sceneEvents.push(
            Events.on(engine, 'beforeUpdate', function(event) {
                counter += 1;

                if (counter === 40)
                    Body.setStatic(bodyG, true);

                if (scaleFactor > 1) {
                    Body.scale(bodyF, scaleFactor, scaleFactor);
                    Body.scale(compound, 0.995, 0.995);

                    // modify bodyE vertices
                    bodyE.vertices[0].x -= 0.2;
                    bodyE.vertices[0].y -= 0.2;
                    bodyE.vertices[1].x += 0.2;
                    bodyE.vertices[1].y -= 0.2;
                    Body.setVertices(bodyE, bodyE.vertices);
                }

                // make bodyA move up and down
                // body is static so must manually update velocity for friction to work
                var py = 300 + 100 * Math.sin(engine.timing.timestamp * 0.002);
                Body.setVelocity(bodyA, { x: 0, y: py - bodyA.position.y });
                Body.setPosition(bodyA, { x: 100, y: py });

                // make compound body move up and down and rotate constantly
                Body.setVelocity(compound, { x: 0, y: py - compound.position.y });
                Body.setAngularVelocity(compound, 0.02);
                Body.setPosition(compound, { x: 600, y: py });
                Body.rotate(compound, 0.02);

                // every 1.5 sec
                if (counter >= 60 * 1.5) {
                    Body.setVelocity(bodyB, { x: 0, y: -10 });
                    Body.setAngle(bodyC, -Math.PI * 0.26);
                    Body.setAngularVelocity(bodyD, 0.2);

                    // reset counter
                    counter = 0;
                    scaleFactor = 1;
                }
            })
        );

        var renderOptions = demo.render.options;
        renderOptions.showAxes = true;
        renderOptions.showCollisions = true;
        renderOptions.showPositions = true;
        renderOptions.showConvexHulls = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Common = Matter.Common;

    Example.mixed = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var stack = Composites.stack(20, 20, 15, 4, 0, 0, function(x, y) {
            var sides = Math.round(Common.random(1, 8));

            // triangles can be a little unstable, so avoid until fixed
            sides = (sides === 3) ? 4 : sides;

            // round the edges of some bodies
            var chamfer = null;
            if (sides > 2 && Common.random() > 0.7) {
                chamfer = {
                    radius: 10
                };
            }

            switch (Math.round(Common.random(0, 1))) {
            case 0:
                if (Common.random() < 0.8) {
                    return Bodies.rectangle(x, y, Common.random(25, 50), Common.random(25, 50), { chamfer: chamfer });
                } else {
                    return Bodies.rectangle(x, y, Common.random(80, 120), Common.random(25, 30), { chamfer: chamfer });
                }
                break;
            case 1:
                return Bodies.polygon(x, y, sides, Common.random(25, 50), { chamfer: chamfer });
            }
        });

        World.add(world, stack);
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Common = Matter.Common;

    Example.mixedSolid = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var stack = Composites.stack(50, 50, 12, 3, 0, 0, function(x, y) {
            switch (Math.round(Common.random(0, 1))) {

            case 0:
                if (Common.random() < 0.8) {
                    return Bodies.rectangle(x, y, Common.random(20, 50), Common.random(20, 50));
                } else {
                    return Bodies.rectangle(x, y, Common.random(80, 120), Common.random(20, 30));
                }
                break;
            case 1:
                return Bodies.polygon(x, y, Math.round(Common.random(1, 8)), Common.random(20, 50));

            }
        });

        World.add(world, stack);

        var renderOptions = demo.render.options;
        renderOptions.wireframes = false;
        renderOptions.showAngleIndicator = false;
    };

})();
(function() {

    var World = Matter.World,
        Body = Matter.Body,
        Composites = Matter.Composites;

    Example.newtonsCradle = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var cradle = Composites.newtonsCradle(280, 100, 5, 30, 200);
        World.add(world, cradle);
        Body.translate(cradle.bodies[0], { x: -180, y: -100 });

        cradle = Composites.newtonsCradle(280, 380, 7, 20, 140);
        World.add(world, cradle);
        Body.translate(cradle.bodies[0], { x: -140, y: -100 });

        var renderOptions = demo.render.options;
        renderOptions.showVelocity = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites;

    Example.pyramid = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var stack = Composites.pyramid(100, 258, 15, 10, 0, 0, function(x, y) {
            return Bodies.rectangle(x, y, 40, 40);
        });

        World.add(world, stack);
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composite = Matter.Composite,
        Composites = Matter.Composites,
        Common = Matter.Common,
        Events = Matter.Events,
        Vertices = Matter.Vertices,
        Query = Matter.Query;

    Example.raycasting = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            sceneEvents = demo.sceneEvents,
            mouseConstraint = demo.mouseConstraint;

        var stack = Composites.stack(20, 20, 15, 4, 0, 0, function(x, y) {
            switch (Math.round(Common.random(0, 1))) {

            case 0:
                if (Common.random() < 0.8) {
                    return Bodies.rectangle(x, y, Common.random(20, 50), Common.random(20, 50));
                } else {
                    return Bodies.rectangle(x, y, Common.random(80, 120), Common.random(20, 30));
                }
                break;
            case 1:
                var sides = Math.round(Common.random(1, 8));
                sides = (sides === 3) ? 4 : sides;
                return Bodies.polygon(x, y, sides, Common.random(20, 50));
            }
        });

        var star = Vertices.fromPath('50 0 63 38 100 38 69 59 82 100 50 75 18 100 31 59 0 38 37 38'),
            concave = Bodies.fromVertices(200, 200, star);

        World.add(world, [stack, concave]);

        sceneEvents.push(
            Events.on(demo.render, 'afterRender', function() {
                var mouse = mouseConstraint.mouse,
                    context = demo.render.context,
                    bodies = Composite.allBodies(engine.world),
                    startPoint = { x: 400, y: 100 },
                    endPoint = mouse.position;

                var collisions = Query.ray(bodies, startPoint, endPoint);

                context.beginPath();
                context.moveTo(startPoint.x, startPoint.y);
                context.lineTo(endPoint.x, endPoint.y);
                if (collisions.length > 0) {
                    context.strokeStyle = '#fff';
                } else {
                    context.strokeStyle = '#555';
                }
                context.lineWidth = 0.5;
                context.stroke();

                for (var i = 0; i < collisions.length; i++) {
                    var collision = collisions[i];
                    context.rect(collision.bodyA.position.x - 4.5, collision.bodyA.position.y - 4.5, 8, 8);
                }

                context.fillStyle = 'rgba(255,165,0,0.7)';
                context.fill();
            })
        );
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies;

    Example.restitution = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var rest = 0.9,
            space = 600 / 5;

        World.add(world, [
            Bodies.rectangle(100 + space * 0, 150, 50, 50, { restitution: rest }),
            Bodies.rectangle(100 + space * 1, 150, 50, 50, { restitution: rest, angle: -Math.PI * 0.15 }),
            Bodies.rectangle(100 + space * 2, 150, 50, 50, { restitution: rest, angle: -Math.PI * 0.25 }),
            Bodies.circle(100 + space * 3, 150, 25, { restitution: rest }),
            Bodies.rectangle(100 + space * 5, 150, 180, 20, { restitution: rest, angle: -Math.PI * 0.5 })
        ]);

        var renderOptions = demo.render.options;
        renderOptions.showCollisions = true;
        renderOptions.showVelocity = true;
        renderOptions.showAngleIndicator = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies;

    Example.rounded = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        World.add(world, [
            Bodies.rectangle(200, 200, 100, 100, {
                chamfer: { radius: 20 }
            }),

            Bodies.rectangle(300, 200, 100, 100, {
                chamfer: { radius: [90, 0, 0, 0] }
            }),

            Bodies.rectangle(400, 200, 200, 200, {
                chamfer: { radius: [150, 20, 40, 20] }
            }),

            Bodies.rectangle(200, 200, 200, 200, {
                chamfer: { radius: [150, 20, 150, 20] }
            }),

            Bodies.rectangle(300, 200, 200, 50, {
                chamfer: { radius: [25, 25, 0, 0] }
            }),

            Bodies.polygon(200, 100, 8, 80, {
                chamfer: { radius: 30 }
            }),

            Bodies.polygon(300, 100, 5, 80, {
                chamfer: { radius: [10, 40, 20, 40, 10] }
            }),

            Bodies.polygon(400, 200, 3, 50, {
                chamfer: { radius: [20, 0, 20] }
            })
        ]);

        var renderOptions = demo.render.options;
        renderOptions.showAxes = true;
        renderOptions.showCollisions = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Common = Matter.Common,
        Events = Matter.Events;

    Example.sensors = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            sceneEvents = demo.sceneEvents;

        var redColor = '#C44D58',
            greenColor = '#C7F464';

        var collider = Bodies.rectangle(400, 300, 500, 50, {
            isSensor: true,
            isStatic: true,
            render: {
                strokeStyle: redColor,
                fillStyle: 'transparent'
            }
        });

        World.add(world, collider);

        World.add(world,
            Bodies.circle(400, 40, 30, {
                render: {
                    strokeStyle: greenColor,
                    fillStyle: 'transparent'
                }
            })
        );

        sceneEvents.push(
            Events.on(engine, 'collisionStart', function(event) {
                var pairs = event.pairs;

                for (var i = 0, j = pairs.length; i != j; ++i) {
                    var pair = pairs[i];

                    if (pair.bodyA === collider) {
                        pair.bodyB.render.strokeStyle = redColor;
                    } else if (pair.bodyB === collider) {
                        pair.bodyA.render.strokeStyle = redColor;
                    }
                }
            }),
            Events.on(engine, 'collisionEnd', function(event) {
                var pairs = event.pairs;

                for (var i = 0, j = pairs.length; i != j; ++i) {
                    var pair = pairs[i];

                    if (pair.bodyA === collider) {
                        pair.bodyB.render.strokeStyle = greenColor;
                    } else if (pair.bodyB === collider) {
                        pair.bodyA.render.strokeStyle = greenColor;
                    }
                }
            })
        );

        var renderOptions = demo.render.options;
        renderOptions.wireframes = false;
        renderOptions.background = '#222';
        renderOptions.showAngleIndicator = false;
    };
})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Common = Matter.Common,
        Events = Matter.Events;

    Example.sleeping = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var stack = Composites.stack(50, 50, 12, 3, 0, 0, function(x, y) {
            switch (Math.round(Common.random(0, 1))) {

            case 0:
                if (Common.random() < 0.8) {
                    return Bodies.rectangle(x, y, Common.random(20, 50), Common.random(20, 50));
                } else {
                    return Bodies.rectangle(x, y, Common.random(80, 120), Common.random(20, 30));
                }
                break;
            case 1:
                return Bodies.polygon(x, y, Math.round(Common.random(1, 8)), Common.random(20, 50));

            }
        });

        World.add(world, stack);

        for (var i = 0; i < stack.bodies.length; i++) {
            Events.on(stack.bodies[i], 'sleepStart sleepEnd', function(event) {
                var body = this;
                console.log('body id', body.id, 'sleeping:', body.isSleeping);
            });
        }

        engine.enableSleeping = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Constraint = Matter.Constraint,
        Events = Matter.Events;

    Example.slingshot = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            mouseConstraint = demo.mouseConstraint;

        world.bodies = [];

        var ground = Bodies.rectangle(395, 600, 815, 50, { isStatic: true, render: { visible: false } }),
            rockOptions = { density: 0.004, render: { sprite: { texture: './img/rock.png' } } },
            rock = Bodies.polygon(170, 450, 8, 20, rockOptions),
            anchor = { x: 170, y: 450 },
            elastic = Constraint.create({
                pointA: anchor,
                bodyB: rock,
                stiffness: 0.05,
                render: {
                    lineWidth: 5,
                    strokeStyle: '#dfa417'
                }
            });

        var pyramid = Composites.pyramid(500, 300, 9, 10, 0, 0, function(x, y, column) {
            var texture = column % 2 === 0 ? './img/block.png' : './img/block-2.png';
            return Bodies.rectangle(x, y, 25, 40, { render: { sprite: { texture: texture } } });
        });

        var ground2 = Bodies.rectangle(610, 250, 200, 20, {
            isStatic: true,
            render: {
                fillStyle: '#edc51e',
                strokeStyle: '#b5a91c'
            }
        });

        var pyramid2 = Composites.pyramid(550, 0, 5, 10, 0, 0, function(x, y, column) {
            var texture = column % 2 === 0 ? './img/block.png' : './img/block-2.png';
            return Bodies.rectangle(x, y, 25, 40, { render: { sprite: { texture: texture } } });
        });

        World.add(engine.world, [ground, pyramid, ground2, pyramid2, rock, elastic]);

        Events.on(engine, 'afterUpdate', function() {
            if (mouseConstraint.mouse.button === -1 && (rock.position.x > 190 || rock.position.y < 430)) {
                rock = Bodies.polygon(170, 450, 7, 20, rockOptions);
                World.add(engine.world, rock);
                elastic.bodyB = rock;
            }
        });

        var renderOptions = demo.render.options;
        renderOptions.wireframes = false;
        renderOptions.showAngleIndicator = false;
        renderOptions.background = './img/background.png';
    };

})();
(function() {

    var World = Matter.World,
        Composites = Matter.Composites;

    Example.softBody = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var particleOptions = {
            friction: 0.05,
            frictionStatic: 0.1,
            render: { visible: true }
        };

        World.add(world, [
            Composites.softBody(250, 100, 5, 5, 0, 0, true, 18, particleOptions),
            Composites.softBody(400, 300, 8, 3, 0, 0, true, 15, particleOptions),
            Composites.softBody(250, 400, 4, 4, 0, 0, true, 15, particleOptions)
        ]);

        var renderOptions = demo.render.options;
        renderOptions.showAngleIndicator = false;
    };

})();

(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Common = Matter.Common;

    Example.sprites = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            offset = 10,
            options = {
                isStatic: true,
                render: {
                    visible: false
                }
            };

        world.bodies = [];

        // these static walls will not be rendered in this sprites example, see options
        World.add(world, [
            Bodies.rectangle(400, -offset, 800.5 + 2 * offset, 50.5, options),
            Bodies.rectangle(400, 600 + offset, 800.5 + 2 * offset, 50.5, options),
            Bodies.rectangle(800 + offset, 300, 50.5, 600.5 + 2 * offset, options),
            Bodies.rectangle(-offset, 300, 50.5, 600.5 + 2 * offset, options)
        ]);

        var stack = Composites.stack(20, 20, 10, 4, 0, 0, function(x, y) {
            if (Common.random() > 0.35) {
                return Bodies.rectangle(x, y, 64, 64, {
                    render: {
                        strokeStyle: '#ffffff',
                        sprite: {
                            texture: './img/box.png'
                        }
                    }
                });
            } else {
                return Bodies.circle(x, y, 46, {
                    density: 0.0005,
                    frictionAir: 0.06,
                    restitution: 0.3,
                    friction: 0.01,
                    render: {
                        sprite: {
                            texture: './img/ball.png'
                        }
                    }
                });
            }
        });

        World.add(world, stack);

        var renderOptions = demo.render.options;
        renderOptions.background = './img/wall-bg.jpg';
        renderOptions.showAngleIndicator = false;
        renderOptions.wireframes = false;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites;

    Example.stack = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var stack = Composites.stack(100, 380, 10, 5, 0, 0, function(x, y) {
            return Bodies.rectangle(x, y, 40, 40);
        });

        World.add(world, stack);
    };

})();

(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Body = Matter.Body,
        Composites = Matter.Composites,
        Events = Matter.Events;

    Example.staticFriction = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            sceneEvents = demo.sceneEvents;

        var body = Bodies.rectangle(400, 500, 200, 60, { isStatic: true, chamfer: 10 }),
            size = 50,
            counter = -1;

        var stack = Composites.stack(350, 470 - 6 * size, 1, 6, 0, 0, function(x, y) {
            return Bodies.rectangle(x, y, size * 2, size, {
                slop: 0.5,
                friction: 1,
                frictionStatic: Infinity
            });
        });

        World.add(world, [body, stack]);

        sceneEvents.push(
            Events.on(engine, 'beforeUpdate', function(event) {
                counter += 0.014;

                if (counter < 0) {
                    return;
                }

                var px = 400 + 100 * Math.sin(counter);

                // body is static so must manually update velocity for friction to work
                Body.setVelocity(body, { x: px - body.position.x, y: 0 });
                Body.setPosition(body, { x: px, y: body.position.y });
            })
        );

        var renderOptions = demo.render.options;
        renderOptions.showAngleIndicator = false;
        renderOptions.showVelocity = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites;

    Example.stress = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var stack = Composites.stack(90, 50, 18, 15, 0, 0, function(x, y) {
            return Bodies.rectangle(x, y, 35, 35);
        });

        World.add(world, stack);

        var renderOptions = demo.render.options;
        renderOptions.showAngleIndicator = false;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites;

    Example.stress2 = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var stack = Composites.stack(100, 120, 25, 18, 0, 0, function(x, y) {
            return Bodies.rectangle(x, y, 25, 25);
        });

        World.add(world, stack);

        var renderOptions = demo.render.options;
        renderOptions.showAngleIndicator = false;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Common = Matter.Common,
        Vertices = Matter.Vertices,
        Svg = Matter.Svg;

    Example.svg = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var svgs = [
            'iconmonstr-check-mark-8-icon',
            'iconmonstr-paperclip-2-icon',
            'iconmonstr-puzzle-icon',
            'iconmonstr-user-icon'
        ];

        for (var i = 0; i < svgs.length; i += 1) {
            (function(i) {
                $.get('./svg/' + svgs[i] + '.svg').done(function(data) {
                    var vertexSets = [],
                        color = Common.choose(['#556270', '#4ECDC4', '#C7F464', '#FF6B6B', '#C44D58']);

                    $(data).find('path').each(function(i, path) {
                        var points = Svg.pathToVertices(path, 30);
                        vertexSets.push(Vertices.scale(points, 0.4, 0.4));
                    });

                    World.add(world, Bodies.fromVertices(100 + i * 150, 200 + i * 50, vertexSets, {
                        render: {
                            fillStyle: color,
                            strokeStyle: color
                        }
                    }, true));
                });
            })(i);
        }

        $.get('./svg/svg.svg').done(function(data) {
            var vertexSets = [],
                color = Common.choose(['#556270', '#4ECDC4', '#C7F464', '#FF6B6B', '#C44D58']);

            $(data).find('path').each(function(i, path) {
                vertexSets.push(Svg.pathToVertices(path, 30));
            });

            World.add(world, Bodies.fromVertices(400, 80, vertexSets, {
                render: {
                    fillStyle: color,
                    strokeStyle: color
                }
            }, true));
        });

        var renderOptions = demo.render.options;
        renderOptions.showAngleIndicator = false;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Common = Matter.Common,
        Query = Matter.Query,
        Svg = Matter.Svg;

    Example.terrain = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        world.bodies = [];

        var terrain;

        $.get('./svg/terrain.svg').done(function(data) {
            var vertexSets = [],
                color = Common.choose(['#556270', '#4ECDC4', '#C7F464', '#FF6B6B', '#C44D58']);

            $(data).find('path').each(function(i, path) {
                vertexSets.push(Svg.pathToVertices(path, 30));
            });

            terrain = Bodies.fromVertices(400, 350, vertexSets, {
                isStatic: true,
                render: {
                    fillStyle: color,
                    strokeStyle: color
                }
            }, true);

            World.add(world, terrain);

            var bodyOptions = {
                frictionAir: 0,
                friction: 0.0001,
                restitution: 0.6
            };

            World.add(world, Composites.stack(80, 100, 20, 20, 10, 10, function(x, y) {
                if (Query.point([terrain], { x: x, y: y }).length === 0) {
                    return Bodies.polygon(x, y, 5, 12, bodyOptions);
                }
            }));
        });

        var renderOptions = demo.render.options;
        renderOptions.showAngleIndicator = false;
        renderOptions.showVelocity = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Body = Matter.Body,
        Composite = Matter.Composite,
        Composites = Matter.Composites,
        Common = Matter.Common,
        Events = Matter.Events;

    Example.timescale = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            sceneEvents = demo.sceneEvents;

        var explosion = function(engine) {
            var bodies = Composite.allBodies(engine.world);

            for (var i = 0; i < bodies.length; i++) {
                var body = bodies[i];

                if (!body.isStatic && body.position.y >= 500) {
                    var forceMagnitude = 0.05 * body.mass;

                    Body.applyForce(body, body.position, {
                        x: (forceMagnitude + Common.random() * forceMagnitude) * Common.choose([1, -1]),
                        y: -forceMagnitude + Common.random() * -forceMagnitude
                    });
                }
            }
        };

        var timeScaleTarget = 1,
            counter = 0;

        sceneEvents.push(
            Events.on(engine, 'afterUpdate', function(event) {
                // tween the timescale for bullet time slow-mo
                engine.timing.timeScale += (timeScaleTarget - engine.timing.timeScale) * 0.05;

                counter += 1;

                // every 1.5 sec
                if (counter >= 60 * 1.5) {

                    // flip the timescale
                    if (timeScaleTarget < 1) {
                        timeScaleTarget = 1;
                    } else {
                        timeScaleTarget = 0.05;
                    }

                    // create some random forces
                    explosion(engine);

                    // reset counter
                    counter = 0;
                }
            })
        );

        var bodyOptions = {
            frictionAir: 0,
            friction: 0.0001,
            restitution: 0.8
        };

        // add some small bouncy circles... remember Swordfish?
        World.add(world, Composites.stack(20, 100, 15, 3, 20, 40, function(x, y) {
            return Bodies.circle(x, y, Common.random(10, 20), bodyOptions);
        }));

        // add some larger random bouncy objects
        World.add(world, Composites.stack(50, 50, 8, 3, 0, 0, function(x, y) {
            switch (Math.round(Common.random(0, 1))) {

            case 0:
                if (Common.random() < 0.8) {
                    return Bodies.rectangle(x, y, Common.random(20, 50), Common.random(20, 50), bodyOptions);
                } else {
                    return Bodies.rectangle(x, y, Common.random(80, 120), Common.random(20, 30), bodyOptions);
                }
                break;
            case 1:
                return Bodies.polygon(x, y, Math.round(Common.random(4, 8)), Common.random(20, 50), bodyOptions);

            }
        }));
    };

})();

(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Common = Matter.Common,
        Events = Matter.Events,
        Bounds = Matter.Bounds,
        Vector = Matter.Vector,
        Mouse = Matter.Mouse;

    Example.views = function(demo) {
        var engine = demo.engine,
            world = engine.world,
            sceneEvents = demo.sceneEvents,
            mouseConstraint = demo.mouseConstraint;

        var stack = Composites.stack(20, 20, 15, 4, 0, 0, function(x, y) {
            switch (Math.round(Common.random(0, 1))) {

            case 0:
                if (Common.random() < 0.8) {
                    return Bodies.rectangle(x, y, Common.random(20, 50), Common.random(20, 50));
                } else {
                    return Bodies.rectangle(x, y, Common.random(80, 120), Common.random(20, 30));
                }
                break;
            case 1:
                var sides = Math.round(Common.random(1, 8));
                sides = (sides === 3) ? 4 : sides;
                return Bodies.polygon(x, y, sides, Common.random(20, 50));
            }
        });

        World.add(world, stack);

        // get the centre of the viewport
        var viewportCentre = {
            x: demo.render.options.width * 0.5,
            y: demo.render.options.height * 0.5
        };

        // make the world bounds a little bigger than the render bounds
        world.bounds.min.x = -300;
        world.bounds.min.y = -300;
        world.bounds.max.x = 1100;
        world.bounds.max.y = 900;

        // keep track of current bounds scale (view zoom)
        var boundsScaleTarget = 1,
            boundsScale = {
                x: 1,
                y: 1
            };

        // use the engine tick event to control our view
        sceneEvents.push(
            Events.on(engine, 'beforeTick', function() {
                var world = engine.world,
                    mouse = mouseConstraint.mouse,
                    render = demo.render,
                    translate;

                // mouse wheel controls zoom
                var scaleFactor = mouse.wheelDelta * -0.1;
                if (scaleFactor !== 0) {
                    if ((scaleFactor < 0 && boundsScale.x >= 0.6) || (scaleFactor > 0 && boundsScale.x <= 1.4)) {
                        boundsScaleTarget += scaleFactor;
                    }
                }

                // if scale has changed
                if (Math.abs(boundsScale.x - boundsScaleTarget) > 0.01) {
                    // smoothly tween scale factor
                    scaleFactor = (boundsScaleTarget - boundsScale.x) * 0.2;
                    boundsScale.x += scaleFactor;
                    boundsScale.y += scaleFactor;

                    // scale the render bounds
                    render.bounds.max.x = render.bounds.min.x + render.options.width * boundsScale.x;
                    render.bounds.max.y = render.bounds.min.y + render.options.height * boundsScale.y;

                    // translate so zoom is from centre of view
                    translate = {
                        x: render.options.width * scaleFactor * -0.5,
                        y: render.options.height * scaleFactor * -0.5
                    };

                    Bounds.translate(render.bounds, translate);

                    // update mouse
                    Mouse.setScale(mouse, boundsScale);
                    Mouse.setOffset(mouse, render.bounds.min);
                }

                // get vector from mouse relative to centre of viewport
                var deltaCentre = Vector.sub(mouse.absolute, viewportCentre),
                    centreDist = Vector.magnitude(deltaCentre);

                // translate the view if mouse has moved over 50px from the centre of viewport
                if (centreDist > 50) {
                    // create a vector to translate the view, allowing the user to control view speed
                    var direction = Vector.normalise(deltaCentre),
                        speed = Math.min(10, Math.pow(centreDist - 50, 2) * 0.0002);

                    translate = Vector.mult(direction, speed);

                    // prevent the view moving outside the world bounds
                    if (render.bounds.min.x + translate.x < world.bounds.min.x)
                        translate.x = world.bounds.min.x - render.bounds.min.x;

                    if (render.bounds.max.x + translate.x > world.bounds.max.x)
                        translate.x = world.bounds.max.x - render.bounds.max.x;

                    if (render.bounds.min.y + translate.y < world.bounds.min.y)
                        translate.y = world.bounds.min.y - render.bounds.min.y;

                    if (render.bounds.max.y + translate.y > world.bounds.max.y)
                        translate.y = world.bounds.max.y - render.bounds.max.y;

                    // move the view
                    Bounds.translate(render.bounds, translate);

                    // we must update the mouse too
                    Mouse.setOffset(mouse, render.bounds.min);
                }
            })
        );

        // must enable renderOptions.hasBounds for views to work
        var renderOptions = demo.render.options;
        renderOptions.hasBounds = true;
    };

})();
(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Constraint = Matter.Constraint;

    Example.wreckingBall = function(demo) {
        var engine = demo.engine,
            world = engine.world;

        var rows = 10,
            yy = 600 - 21 - 40 * rows;

        var stack = Composites.stack(400, yy, 5, rows, 0, 0, function(x, y) {
            return Bodies.rectangle(x, y, 40, 40);
        });

        World.add(world, stack);

        var ball = Bodies.circle(100, 400, 50, { density: 0.04, frictionAir: 0.005});

        World.add(world, ball);
        World.add(world, Constraint.create({
            pointA: { x: 300, y: 100 },
            bodyB: ball
        }));
    };

////////////////////////////////////////////////////////////////////////////////
    // Michael's Worlds
////////////////////////////////////////////////////////////////////////////////

(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Body = Matter.Body;;

    Example.simpleHockey = function(demo) {

        var engine = demo.engine,
            world = engine.world;

        engine.world.gravity.y = 0;
        engine.world.gravity.x = 0;

        // now, can I make it generate random initial locations and velocities?
        var num_obj = 3
        // use demo.w_offset, demo.w_cx, demo.w_cy

        world.bounds = {
                min: { x: 0, y: 0 },
                max: { x: 2*demo.w_cx, y: 2*demo.w_cy }
            }
        console.log(world.bounds)

        var stack = Composites.stack(0, 100, 5, 1, 20, 0, function(x, y) {
            var body = Bodies.circle(x, y, 50, { restitution: 1,
                                             friction: 0,
                                             frictionAir: 0,
                                             frictionStatic: 0,
                                             inertia: Infinity,
                                             inverseInertia: 0});  // I initialize the velocity, but they don't continue?
            // Body.setVelocity(body, {x: 5, y: 5});  // this takes out a ball?  If you have more balls than can fit, then this will start moving the balls before all of them fit!
            return body
        });
        World.add(world, stack);

        // doing it this way makes it so that all the balls have initial positions first, and then they move
        engine.world.composites[0].bodies.map(function(elem) {
            Body.setVelocity(elem,{x: 5, y: 5});
        })
    };

})();

(function() {

    var World = Matter.World,
        Bodies = Matter.Bodies,
        Composites = Matter.Composites,
        Body = Matter.Body;

    Example.hockey = function(demo) {

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

        // I could have something like RBD.create() (Rigid Body Dynamics)
        var Hockey = {}

        Hockey.create = function(options) {
            var self = {}; // instance of the Hockey class

            // these should not be mutated
            var params = {show: false,
                          num_obj: 10,
                          cx: 400,
                          cy: 300,
                          max_v0: 20,
                          obj_radius: 50 };
            self.params = params;

            // var engine = Engine.create();
            var engine = demo.engine;
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
                // console.log('object', i)

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
            }

            // set velocities
            _.each(_.zip(self.engine.world.bodies
                        .filter(function(elem) {
                                    return elem.label === 'Circle Body';
                                }), self.v0),
                        function(pair){
                            Body.setVelocity(pair[0],pair[1]); // pair[1] is object, pair[2] is velocity
                        });
        };

        var hockey = Hockey.create()
        Hockey.init(hockey)

        })();

    };

})();

})();
