use eyre;
use fenris::assembly::global::{apply_homogeneous_dirichlet_bc_csr, color_nodes, CsrParAssembler};
use fenris::assembly::local::ElementEllipticAssemblerBuilder;
use fenris::mesh::procedural::create_unit_box_uniform_tet_mesh_3d;
use fenris::nalgebra::DVector;
use fenris::quadrature::CanonicalStiffnessQuadrature;
use fenris_solid::materials::{LameParameters, LinearElasticMaterial, YoungPoisson};
use fenris_solid::MaterialEllipticOperator;
use nalgebra_sparse::io::save_to_matrix_market_file;
use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(about)]
struct Cli {
    #[arg(short, long, default_value_t = 3)]
    resolution: usize,

    #[arg(short, long, default_value = "output/stiffness_matrix.mm")]
    output: PathBuf,
}

fn main() -> eyre::Result<()> {
    let cli = Cli::parse();

    let material = LinearElasticMaterial;
    let operator = MaterialEllipticOperator::new(&material);
    let lame_parameters = LameParameters::from(YoungPoisson {
        young: 1e7,
        poisson: 0.4,
    });

    let mesh = create_unit_box_uniform_tet_mesh_3d(cli.resolution);

    // This is not needed for linear problems, but the API is designed for more general
    // non-linear operators ATM so we still need to provide some dummy displacement value.
    let u = DVector::zeros(3 * mesh.vertices().len());
    let qtable = mesh
        .canonical_stiffness_quadrature()
        .with_uniform_data(lame_parameters);
    let element_assembler = ElementEllipticAssemblerBuilder::new()
        .with_finite_element_space(&mesh)
        .with_operator(&operator)
        .with_quadrature_table(&qtable)
        .with_u(&u)
        .build();

    let colors = color_nodes(&mesh);
    let mut stiffness_matrix = CsrParAssembler::default().assemble(&colors, &element_assembler)?;

    let boundary_nodes = mesh.find_boundary_vertices();
    apply_homogeneous_dirichlet_bc_csr(&mut stiffness_matrix, &boundary_nodes, 3);

    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    save_to_matrix_market_file(&stiffness_matrix, &cli.output)?;

    println!(
        "Assembled and exported {n} x {n} stiffness matrix with {nnz} non-zeros",
        n = stiffness_matrix.nrows(),
        nnz = stiffness_matrix.nnz()
    );

    Ok(())
}
