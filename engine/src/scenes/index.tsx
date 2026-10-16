import React from 'react';
import {Kicker} from '../components';
import {useTheme} from '../theme';
import {useVideo} from '../context';
import {norm} from '../timeline';
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
  const {look} = useTheme();
  const {lines} = useVideo();
  const scene = line.scene;
  const Comp = REGISTRY[scene?.type];
  if (!Comp) return null;
  // Kickers are rare: only when set, never on title or outro, never twice in a row, and not on the cobalt stat block.
  const prev = lines[line.index - 1]?.kicker;
  const show = !!line.kicker && scene.type !== 'title' && scene.type !== 'outro' && norm(line.kicker) !== norm(prev ?? '') && !(look === 'cobalt' && scene.type === 'stat');
  return (
    <>
      <Comp scene={scene} />
      {show && <Kicker>{line.kicker}</Kicker>}
    </>
  );
};
