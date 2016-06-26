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
        var CircularJSON = require('circular-json')
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
    Demo.run = function(json_data) {

        //TODO: note that here you should load the demo engine with the json file

        var demo = {}
        demo.engine = Engine.create()
        demo.runner = Engine.run(demo.engine)
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

        var trajectories = json_data[0]  // extra 0 for batch mode
        var num_obj = trajectories.length
        var num_steps = trajectories[0].length

        var i = 0
        function f() {
            console.log( i );
            var entities = Composite.allBodies(demo.engine.world)
                .filter(function(elem) {
                            return elem.label === 'Entity';
                        })
            var entity_ids = entities.map(function(elem) {
                                return elem.id});

            for (id = 0; id < entity_ids.length; id++) { //id = 0 corresponds to world!
                var body = Composite.get(demo.engine.world, entity_ids[id], 'body')
                // set the position here
                Body.setPosition(body, trajectories[id][i].position)
            }

            Runner.tick(demo.runner, demo.engine);
            i++;
            if( i < num_steps ){
                setTimeout( f, 1000 );
            }
        }
        f();

    }

    // call init when the page has loaded fully
    if (!_isAutomatedTest) {
        window.loadFile = function loadFile(file){
            var fr = new FileReader();
            fr.onload = function(){
                Demo.run(window.CircularJSON.parse(fr.result))
            }
            fr.readAsText(file)
        }
    }
})();
