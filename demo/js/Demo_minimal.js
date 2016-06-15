/**
* The Matter.js demo page controller and example runner.
*
* NOTE: For the actual example code, refer to the source files in `/examples/`.
*
* @class Demo
*/

(function() {

    var _isBrowser = typeof window !== 'undefined' && window.location,
        _useInspector = _isBrowser && window.location.hash.indexOf('-inspect') !== -1,
        _isMobile = _isBrowser && /(ipad|iphone|ipod|android)/gi.test(navigator.userAgent),
        _isAutomatedTest = !_isBrowser || window._phantom;

    // var Matter = _isBrowser ? window.Matter : require('../../build/matter-dev.js');
    var Matter = _isBrowser ? window.Matter : require('matter-js');

    var Demo = {};
    Matter.Demo = Demo;

    if (!_isBrowser) {
        var jsonfile = require('jsonfile')
        var assert = require('assert')
        var utils = require('../../utils')
        var sleep = require('sleep')
        require('./Examples')
        var env = process.argv.slice(2)[0]
        if (env == null)
            throw('Please provide an enviornment, e.g. node Demo.js hockey')
        module.exports = Demo;
        window = {};
    }

    // Matter aliases
    var Body = Matter.Body,
        Example = Matter.Example,
        Engine = Matter.Engine,
        World = Matter.World,
        Common = Matter.Common,
        Composite = Matter.Composite,
        Bodies = Matter.Bodies,
        Events = Matter.Events,
        Runner = Matter.Runner,
        Render = Matter.Render;

    // Create the engine
    Demo.run = function() {
        var demo = {}
        demo.engine = Engine.create()
        demo.runner = Engine.run(demo.engine)  // should I do this?
        demo.container = document.getElementById('canvas-container');
        demo.render = Render.create({element: demo.container, engine: demo.engine})
        Render.run(demo.render)

        demo.w_offset = 5;  // world offset
        demo.w_cx = 400;
        demo.w_cy = 300;

        var world_border = Composite.create({label:'Border'});

        Composite.add(world_border, [
            Bodies.rectangle(demo.w_cx, -demo.w_offset, 2*demo.w_cx + 2*demo.w_offset, 2*demo.w_offset, { isStatic: true, restitution: 1 }),
            Bodies.rectangle(demo.w_cx, 600+demo.w_offset, 2*demo.w_cx + 2*demo.w_offset, 2*demo.w_offset, { isStatic: true, restitution: 1 }),
            Bodies.rectangle(2*demo.w_cx + demo.w_offset, demo.w_cy, 2*demo.w_offset, 2*demo.w_cy + 2*demo.w_offset, { isStatic: true, restitution: 1 }),
            Bodies.rectangle(-demo.w_offset, demo.w_cy, 2*demo.w_offset, 2*demo.w_cy + 2*demo.w_offset, { isStatic: true, restitution: 1 })
        ]);

        World.add(demo.engine.world, world_border)  // its parent is a circular reference!

        var sceneName = 'm_balls'
        Example[sceneName](demo);



        // Ok, now let's manually update
        Runner.stop(demo.runner)

        var i = 0, howManyTimes = 100;
        function f() {
            console.log( i );

            // here you can manually set the postion.
            // Let's try it. Let's manually reset the position
            Runner.tick(demo.runner, demo.engine);
            i++;
            if( true ){  // here you could replace true with a stopping condition
                setTimeout( f, 1000 );
            }
        }
        f();

    }

    // call init when the page has loaded fully
    if (!_isAutomatedTest) {
        if (window.addEventListener) {
            window.addEventListener('load', Demo.run);
        } else if (window.attachEvent) {
            window.attachEvent('load', Demo.run);
        }
    }
})();
