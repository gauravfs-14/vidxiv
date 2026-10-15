import React from 'react';
import {Kicker} from '../components';
import {useTheme} from '../theme';
import type {LineProps, Scene} from '../types';
import {Hook} from './Hook';
import {Title} from './Title';
import {Concept} from './Concept';
import {Steps} from './Steps';
import {FigureScene} from './Figure';
import {Stat} from './Stat';
import {Bars} from './Bars';
import {Compare} from './Compare';
import {Equation} from './Equation';
import {Quote} from './Quote';
import {Outro} from './Outro';

/** Scene type -> component. To add a scene type, add it here, in types.ts, and in sfx.ts. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const REGISTRY: Record<Scene['type'], React.FC<{scene: any}>> = {hook: Hook, title: Title, concept: Concept, steps: Steps, figure: FigureScene, stat: Stat, bars: Bars, compare: Compare, equation: Equation, quote: Quote, outro: Outro};

export const SceneView: React.FC<{line: LineProps}> = ({line}) => {
  const {C} = useTheme();
  const scene = line.scene;
  const Comp = REGISTRY[scene?.type];
  if (!Comp) return null;
  const noKicker = scene.type === 'title' || scene.type === 'outro';
  return (
    <>
      <Comp scene={scene} />
      {line.kicker && !noKicker && <Kicker color={C.accent}>{line.kicker}</Kicker>}
    </>
  );
};
